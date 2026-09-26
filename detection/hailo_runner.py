"""Hailo-8 inference runners.

Hailo-8 同一时刻只支持一个激活的 VDevice, 一个激活的 NetworkGroup.
当需要在同一进程跑多个 HEF 时, 必须共享 VDevice 并按需 swap activation.

本模块提供两个类:
    HailoRunner        - 独占 VDevice, 跑单个 HEF. 适合单模型场景.
    HailoMultiRunner   - 共享 VDevice, 多个 HEF 装入后按需切换 activation.
                         实测 swap 延迟 < 2ms.

激活的 InferVStreams pipeline 在整个生命周期内复用, 避免每帧重建 context
带来的 FPS 崩塌.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _import_hailo():
    """延迟导入 HailoRT, 避免在非 Hailo 环境导入报错."""
    from hailo_platform import (  # type: ignore
        HEF,
        VDevice,
        FormatType,
        HailoStreamInterface,
        ConfigureParams,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
    )
    return {
        "HEF": HEF,
        "VDevice": VDevice,
        "FormatType": FormatType,
        "HailoStreamInterface": HailoStreamInterface,
        "ConfigureParams": ConfigureParams,
        "InferVStreams": InferVStreams,
        "InputVStreamParams": InputVStreamParams,
        "OutputVStreamParams": OutputVStreamParams,
    }


def _hw_backend() -> str:
    """探测接口: PCIe (reComputer M.2) 或 Ethernet."""
    return "PCIe"


class HailoRunner:
    """单 HEF 推理包装 (独占 VDevice)."""

    def __init__(
        self,
        hef_path: str,
        input_format: str = "UINT8",
        output_format: str = "FLOAT32",
        interface: Optional[str] = None,
        warmup_runs: int = 3,
    ) -> None:
        self.hef_path = hef_path
        self.input_format = input_format
        self.output_format = output_format
        self.interface = interface or _hw_backend()
        self.warmup_runs = max(1, int(warmup_runs))

        self._vdevice: Any = None
        self._hef: Any = None
        self._network_group: Any = None
        self._ng_params: Any = None
        self._activation: Any = None
        self._pipe_ctx: Any = None
        _pipe: Any = None  # type: ignore[assignment]
        self._infer_pipe = _pipe
        self._input_infos: List[Any] = []
        self._output_infos: List[Any] = []
        self._closed = True

    def __enter__(self) -> "HailoRunner":
        h = _import_hailo()
        iface = h["HailoStreamInterface"].PCIe if self.interface == "PCIe" else None
        if iface is None:
            raise ValueError(f"Unsupported interface: {self.interface}")
        in_fmt = h["FormatType"].UINT8 if self.input_format == "UINT8" else h["FormatType"].FLOAT32
        out_fmt = h["FormatType"].UINT8 if self.output_format == "UINT8" else h["FormatType"].FLOAT32

        self._vdevice = h["VDevice"]()
        self._hef = h["HEF"](self.hef_path)
        cfg = h["ConfigureParams"].create_from_hef(hef=self._hef, interface=iface)
        self._network_group = self._vdevice.configure(self._hef, cfg)[0]
        self._ng_params = self._network_group.create_params()

        self._input_infos = list(self._hef.get_input_vstream_infos())
        self._output_infos = list(self._hef.get_output_vstream_infos())

        in_p = h["InputVStreamParams"].make(self._network_group, format_type=in_fmt)
        out_p = h["OutputVStreamParams"].make(self._network_group, format_type=out_fmt)

        self._activation = self._network_group.activate(self._ng_params)
        self._activation.__enter__()
        self._pipe_ctx = h["InferVStreams"](self._network_group, in_p, out_p)
        self._infer_pipe = self._pipe_ctx.__enter__()
        self._closed = False

        self._warmup()
        logger.info("HailoRunner ready: %s", self.hef_path)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _warmup(self) -> None:
        h = _import_hailo()
        for info in self._input_infos:
            shape = tuple(info.shape)
            if len(shape) >= 3:
                dummy_shape = (1,) + shape
            else:
                dummy_shape = (1, shape[0])
            if self.input_format == "UINT8":
                dummy = np.random.randint(0, 255, dummy_shape, dtype=np.uint8)
            else:
                dummy = np.random.uniform(-1.0, 1.0, dummy_shape).astype(np.float32)
            for _ in range(self.warmup_runs):
                self._infer_pipe.infer({info.name: dummy})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._pipe_ctx is not None:
                self._pipe_ctx.__exit__(None, None, None)
        except Exception:
            logger.exception("HailoRunner: pipe close failed")
        try:
            if self._activation is not None:
                self._activation.__exit__(None, None, None)
        except Exception:
            logger.exception("HailoRunner: activation close failed")
        try:
            if self._vdevice is not None:
                self._vdevice.release()
        except Exception:
            logger.exception("HailoRunner: vdevice release failed")
        self._infer_pipe = None
        self._pipe_ctx = None
        self._activation = None
        self._vdevice = None

    @property
    def input_infos(self) -> List[Any]:
        return list(self._input_infos)

    @property
    def output_infos(self) -> List[Any]:
        return list(self._output_infos)

    def run(self, inputs: Mapping[str, np.ndarray]) -> Dict[str, Any]:
        if self._infer_pipe is None:
            raise RuntimeError("HailoRunner not active")
        return self._infer_pipe.infer(dict(inputs))

    def run_with_timing(self, inputs: Mapping[str, np.ndarray]) -> Tuple[Dict[str, Any], float]:
        t0 = time.perf_counter()
        out = self.run(inputs)
        return out, (time.perf_counter() - t0) * 1000.0


class _Slot:
    """HailoMultiRunner 中每个 HEF 占一个 slot."""

    __slots__ = (
        "name",
        "hef",
        "network_group",
        "in_fmt",
        "out_fmt",
        "ng_params",
        "input_infos",
        "output_infos",
        "activation",
        "pipe_ctx",
        "infer_pipe",
        "is_active",
        "warmed",
    )

    def __init__(self, name: str, hef: Any, network_group: Any, in_fmt: Any, out_fmt: Any) -> None:
        self.name = name
        self.hef = hef
        self.network_group = network_group
        self.in_fmt = in_fmt
        self.out_fmt = out_fmt
        self.ng_params = None
        self.input_infos: List[Any] = []
        self.output_infos: List[Any] = []
        self.activation: Any = None
        self.pipe_ctx: Any = None
        self.infer_pipe: Any = None
        self.is_active = False
        self.warmed = False


class HailoMultiRunner:
    """多 HEF 共享 VDevice.

    Hailo-8 一张卡同时只能激活一个 NetworkGroup. 多个 HEF 在
    ``configure()`` 阶段全部装入, 推理时按需 ``activate(name)`` 切换,
    切换延迟实测 < 2ms.

    用法:
        >>> with HailoMultiRunner() as mr:
        ...     mr.add("yolov8n", "/path/yolov8n.hef")
        ...     mr.add("person_attr", "/path/attr.hef")
        ...     mr.activate("yolov8n")
        ...     out = mr.run("yolov8n", {layer: x})
        ...     mr.activate("person_attr")
        ...     out = mr.run("person_attr", {layer: x})
    """

    def __init__(self, interface: Optional[str] = None, warmup_runs: int = 3) -> None:
        self.interface = interface or _hw_backend()
        self.warmup_runs = max(1, int(warmup_runs))
        self._vdevice: Any = None
        self._slots: Dict[str, _Slot] = {}
        self._active: Optional[str] = None

    def __enter__(self) -> "HailoMultiRunner":
        h = _import_hailo()
        self._vdevice = h["VDevice"]()
        logger.info("HailoMultiRunner: VDevice opened")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def add(
        self,
        name: str,
        hef_path: str,
        input_format: str = "UINT8",
        output_format: str = "FLOAT32",
    ) -> _Slot:
        if self._vdevice is None:
            raise RuntimeError("HailoMultiRunner not entered")
        if name in self._slots:
            raise ValueError(f"Model {name!r} already added")
        h = _import_hailo()
        iface = h["HailoStreamInterface"].PCIe if self.interface == "PCIe" else None
        if iface is None:
            raise ValueError(f"Unsupported interface: {self.interface}")
        in_fmt = h["FormatType"].UINT8 if input_format == "UINT8" else h["FormatType"].FLOAT32
        out_fmt = h["FormatType"].UINT8 if output_format == "UINT8" else h["FormatType"].FLOAT32

        hef = h["HEF"](hef_path)
        cfg = h["ConfigureParams"].create_from_hef(hef=hef, interface=iface)
        ng = self._vdevice.configure(hef, cfg)[0]
        slot = _Slot(name=name, hef=hef, network_group=ng, in_fmt=in_fmt, out_fmt=out_fmt)
        slot.ng_params = ng.create_params()
        slot.input_infos = list(hef.get_input_vstream_infos())
        slot.output_infos = list(hef.get_output_vstream_infos())
        self._slots[name] = slot
        logger.info("HailoMultiRunner: added model %r from %s", name, hef_path)
        return slot

    def activate(self, name: str) -> None:
        if name not in self._slots:
            raise KeyError(f"Model {name!r} not loaded")
        if self._active == name:
            return
        if self._active is not None:
            self._deactivate(self._active)
        slot = self._slots[name]
        h = _import_hailo()
        slot.activation = slot.network_group.activate(slot.ng_params)
        slot.activation.__enter__()
        in_p = h["InputVStreamParams"].make(slot.network_group, format_type=slot.in_fmt)
        out_p = h["OutputVStreamParams"].make(slot.network_group, format_type=slot.out_fmt)
        slot.pipe_ctx = h["InferVStreams"](slot.network_group, in_p, out_p)
        slot.infer_pipe = slot.pipe_ctx.__enter__()
        slot.is_active = True
        if not slot.warmed:
            self._warmup(slot)
            slot.warmed = True
        self._active = name

    def _deactivate(self, name: str) -> None:
        slot = self._slots[name]
        if not slot.is_active or slot.activation is None:
            return
        try:
            if slot.pipe_ctx is not None:
                slot.pipe_ctx.__exit__(None, None, None)
        except Exception:
            logger.exception("HailoMultiRunner: pipe close failed for %s", name)
        try:
            slot.activation.__exit__(None, None, None)
        except Exception:
            logger.exception("HailoMultiRunner: activation close failed for %s", name)
        slot.is_active = False
        slot.pipe_ctx = None
        slot.activation = None
        slot.infer_pipe = None

    def _warmup(self, slot: _Slot) -> None:
        h = _import_hailo()
        for info in slot.input_infos:
            shape = tuple(info.shape)
            dummy_shape = (1,) + shape if len(shape) >= 3 else (1, shape[0])
            if slot.in_fmt == h["FormatType"].UINT8:
                dummy = np.random.randint(0, 255, dummy_shape, dtype=np.uint8)
            else:
                dummy = np.random.uniform(-1.0, 1.0, dummy_shape).astype(np.float32)
            for _ in range(self.warmup_runs):
                slot.infer_pipe.infer({info.name: dummy})
        logger.info("HailoMultiRunner: warmed up %s", slot.name)

    def run(self, name: str, inputs: Mapping[str, np.ndarray]) -> Dict[str, Any]:
        if self._active != name:
            self.activate(name)
        slot = self._slots[name]
        if slot.infer_pipe is None:
            raise RuntimeError(f"Slot {name!r} not active")
        return slot.infer_pipe.infer(dict(inputs))

    def run_with_timing(
        self, name: str, inputs: Mapping[str, np.ndarray]
    ) -> Tuple[Dict[str, Any], float]:
        t0 = time.perf_counter()
        out = self.run(name, inputs)
        return out, (time.perf_counter() - t0) * 1000.0

    def get_slot(self, name: str) -> _Slot:
        if name not in self._slots:
            raise KeyError(f"Model {name!r} not loaded")
        return self._slots[name]

    @property
    def active(self) -> Optional[str]:
        return self._active

    @property
    def names(self) -> List[str]:
        return list(self._slots.keys())

    def close(self) -> None:
        for name in list(self._slots.keys()):
            slot = self._slots[name]
            if slot.is_active:
                try:
                    if slot.pipe_ctx is not None:
                        slot.pipe_ctx.__exit__(None, None, None)
                except Exception:
                    logger.exception("HailoMultiRunner: pipe close failed for %s", name)
                try:
                    if slot.activation is not None:
                        slot.activation.__exit__(None, None, None)
                except Exception:
                    logger.exception("HailoMultiRunner: activation close failed for %s", name)
            slot.is_active = False
            slot.pipe_ctx = None
            slot.activation = None
            slot.infer_pipe = None
        self._slots.clear()
        if self._vdevice is not None:
            try:
                self._vdevice.release()
            except Exception:
                logger.exception("HailoMultiRunner: vdevice release failed")
            self._vdevice = None
        self._active = None
        logger.info("HailoMultiRunner: closed")

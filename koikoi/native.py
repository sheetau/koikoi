"""CPU C ABI and header-free NVRTC/CUDA Driver API backend."""

from __future__ import annotations

import ctypes as C
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import torch

SOURCE = Path(__file__).parent / "native"
BUILD = Path(__file__).resolve().parents[1] / "build"


def build_cpu():
    BUILD.mkdir(exist_ok=True)
    digest = hashlib.sha256(
        (SOURCE / "core.h").read_bytes() + (SOURCE / "cpu.cpp").read_bytes()
    ).hexdigest()[:16]
    out = BUILD / (f"koikoi_{digest}.dll" if os.name == "nt" else f"koikoi_{digest}.so")
    if out.exists():
        return out
    if os.name == "nt":
        finder = (
            Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
            / "Microsoft Visual Studio/Installer/vswhere.exe"
        )
        vs = subprocess.check_output(
            [
                str(finder),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            text=True,
        ).strip()
        if not vs:
            raise RuntimeError("Install Visual Studio C++ build tools")
        vcvars = Path(vs) / "VC/Auxiliary/Build/vcvars64.bat"
        script = BUILD / "compile.cmd"
        # All interpolated paths are local, resolved filenames, not user shell input.
        script.write_text(
            f'@echo off\ncall "{vcvars}" >nul\nif errorlevel 1 exit /b 1\n'
            f'cl /nologo /O2 /std:c++17 /EHsc /LD "{SOURCE / "cpu.cpp"}" /Fe:"{out}"\n',
            encoding="utf-8",
        )
        subprocess.run(
            ["cmd", "/d", "/c", str(script)], cwd=BUILD, check=True, capture_output=True
        )
    else:
        subprocess.run(
            [
                "c++",
                "-O3",
                "-std=c++17",
                "-fPIC",
                "-shared",
                str(SOURCE / "cpu.cpp"),
                "-o",
                str(out),
            ],
            check=True,
        )
    return out


class CPU:
    def __init__(self):
        self.lib = C.CDLL(str(build_cpu()))
        for name, signature in {
            "kk_init": [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int],
            "kk_step": [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int],
            "kk_observe": [C.c_void_p] * 4 + [C.c_int],
            "kk_reset": [C.c_void_p, C.c_int],
        }.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = signature, None

    def call(self, name, n, *args):
        getattr(self.lib, name)(
            *[
                C.c_void_p(x.data_ptr()) if isinstance(x, torch.Tensor) else x
                for x in args
            ]
        )


class CUDA:
    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(self.device)
        torch.cuda.init()
        libdir = Path(torch.__file__).parent / "lib"
        if os.name == "nt":
            self.dll_dir = os.add_dll_directory(str(libdir))
            paths = [p for p in libdir.glob("nvrtc64*.dll") if ".alt." not in p.name]
            if not paths:
                raise RuntimeError("CUDA PyTorch wheel does not include NVRTC")
            self.nvrtc = C.CDLL(str(paths[0]))
            self.driver = C.WinDLL("nvcuda.dll")
        else:
            import glob

            candidates = glob.glob(
                str(
                    Path(sys.prefix)
                    / "lib/python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so*"
                )
            )
            self.nvrtc = C.CDLL(candidates[0] if candidates else "libnvrtc.so")
            self.driver = C.CDLL("libcuda.so.1")
        self._bind()
        source = "#define KK_HD __device__ __forceinline__\n" + (
            SOURCE / "core.h"
        ).read_text(encoding="utf-8")
        source += (SOURCE / "kernels.cu").read_text(encoding="utf-8")
        program = C.c_void_p()
        self.check(
            self.nvrtc.nvrtcCreateProgram(
                C.byref(program), source.encode(), b"koikoi.cu", 0, None, None
            ),
            "nvrtcCreateProgram",
        )
        major, minor = torch.cuda.get_device_capability(self.device)
        options = (C.c_char_p * 2)(
            f"--gpu-architecture=compute_{major}{minor}".encode(), b"--std=c++17"
        )
        try:
            error = self.nvrtc.nvrtcCompileProgram(program, 2, options)
            if error:
                size = C.c_size_t()
                self.nvrtc.nvrtcGetProgramLogSize(program, C.byref(size))
                log = C.create_string_buffer(size.value)
                self.nvrtc.nvrtcGetProgramLog(program, log)
                raise RuntimeError(log.value.decode())
            size = C.c_size_t()
            self.check(self.nvrtc.nvrtcGetPTXSize(program, C.byref(size)), "PTX size")
            ptx = C.create_string_buffer(size.value)
            self.check(self.nvrtc.nvrtcGetPTX(program, ptx), "PTX")
        finally:
            self.nvrtc.nvrtcDestroyProgram(C.byref(program))
        self.module = C.c_void_p()
        self.check(
            self.driver.cuModuleLoadData(C.byref(self.module), ptx), "cuModuleLoadData"
        )
        self.functions = {}
        for name in ("kk_init", "kk_step", "kk_observe", "kk_reset"):
            fn = C.c_void_p()
            self.check(
                self.driver.cuModuleGetFunction(
                    C.byref(fn), self.module, name.encode()
                ),
                name,
            )
            self.functions[name] = fn

    def _bind(self):
        specs = {
            "nvrtcCreateProgram": [
                C.c_void_p,
                C.c_char_p,
                C.c_char_p,
                C.c_int,
                C.c_void_p,
                C.c_void_p,
            ],
            "nvrtcCompileProgram": [C.c_void_p, C.c_int, C.c_void_p],
            "nvrtcGetProgramLogSize": [C.c_void_p, C.c_void_p],
            "nvrtcGetProgramLog": [C.c_void_p, C.c_void_p],
            "nvrtcGetPTXSize": [C.c_void_p, C.c_void_p],
            "nvrtcGetPTX": [C.c_void_p, C.c_void_p],
            "nvrtcDestroyProgram": [C.c_void_p],
        }
        for name, args in specs.items():
            fn = getattr(self.nvrtc, name)
            fn.argtypes, fn.restype = args, C.c_int
        for name, args in {
            "cuModuleLoadData": [C.c_void_p, C.c_void_p],
            "cuModuleGetFunction": [C.c_void_p, C.c_void_p, C.c_char_p],
            "cuLaunchKernel": [C.c_void_p] + [C.c_uint] * 7 + [C.c_void_p] * 3,
        }.items():
            fn = getattr(self.driver, name)
            fn.argtypes, fn.restype = args, C.c_int

    @staticmethod
    def check(code, operation):
        if code:
            raise RuntimeError(f"{operation} failed: CUDA/NVRTC error {code}")

    def call(self, name, n, *args):
        with torch.cuda.device(self.device):
            values = [
                C.c_void_p(x.data_ptr()) if isinstance(x, torch.Tensor) else C.c_int(x)
                for x in args
            ]
            params = (C.c_void_p * len(values))(
                *[C.cast(C.byref(x), C.c_void_p) for x in values]
            )
            stream = torch.cuda.current_stream(self.device).cuda_stream
            self.check(
                self.driver.cuLaunchKernel(
                    self.functions[name],
                    (n + 127) // 128,
                    1,
                    1,
                    128,
                    1,
                    1,
                    0,
                    C.c_void_p(stream),
                    params,
                    None,
                ),
                name,
            )

"""Isolated CPU/CUDA compilation of the shared research-variant transitions."""

import ctypes as C
import hashlib
import os
import subprocess
from pathlib import Path

import torch

from .layout import STATE_SIZE
from .native import BUILD, CUDA, SOURCE
from .public_stock import PROFILES as VARIANTS


def source(cuda=False):
    body = "#define KK_HD __device__ __forceinline__\n" if cuda else ""
    body += (
        "namespace kkbase {\n"
        + (SOURCE / "core.h").read_text(encoding="utf-8")
        + "\n}\n"
    )
    body += (SOURCE / "public_stock.h").read_text(encoding="utf-8")
    if cuda:
        body += """
extern "C" __global__ void kp_init(int *s,const int *seeds,int n,int flags,int rounds,int bank){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)publicstock::init_variant(s+i*208,seeds[i],0,flags,rounds,bank);}
extern "C" __global__ void kp_step(int *s,const int *a,int *r,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)publicstock::step_variant(s+i*208,a[i],r+4*i);}
extern "C" __global__ void kp_observe(const int *s,float *o,bool *m,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)publicstock::observation(s+i*208,o+i*660,m+i*50);}
extern "C" __global__ void kp_reset(int *s,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int *q=s+i*208;
 if(q[99]==4)publicstock::init_variant(q,q[110],q[111]+1,q[112],q[104],q[105]);}}
extern "C" __global__ void kp_solved(const int *s,int *a,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)a[i]=publicstock::solved(s+i*208);}
"""
    else:
        export = "__declspec(dllexport)" if os.name == "nt" else ""
        body += f"""
extern "C" {export} void kp_init(int *s,const int *seeds,int n,int flags,int rounds,int bank){{
 for(int i=0;i<n;++i)publicstock::init_variant(s+i*208,seeds[i],0,flags,rounds,bank);}}
extern "C" {export} void kp_step(int *s,const int *a,int *r,int n){{
 for(int i=0;i<n;++i)publicstock::step_variant(s+i*208,a[i],r+4*i);}}
extern "C" {export} void kp_observe(const int *s,float *o,bool *m,int n){{
 for(int i=0;i<n;++i)publicstock::observation(s+i*208,o+i*660,m+i*50);}}
extern "C" {export} void kp_reset(int *s,int n){{
 for(int i=0;i<n;++i){{int *q=s+i*208;
 if(q[99]==4)publicstock::init_variant(q,q[110],q[111]+1,q[112],q[104],q[105]);}}}}
extern "C" {export} void kp_solved(const int *s,int *a,int n){{
 for(int i=0;i<n;++i)a[i]=publicstock::solved(s+i*208);}}
"""
    return body


def build():
    text = source()
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    BUILD.mkdir(exist_ok=True)
    out = BUILD / f"publicstock_{digest}{'.dll' if os.name == 'nt' else '.so'}"
    if out.exists():
        return out
    cpp = BUILD / f"publicstock_{digest}.cpp"
    cpp.write_text(text, encoding="utf-8", newline="\n")
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
        vcvars = Path(vs) / "VC/Auxiliary/Build/vcvars64.bat"
        cmd = BUILD / f"publicstock_{digest}.cmd"
        cmd.write_text(
            f'@echo off\ncall "{vcvars}" >nul\nif errorlevel 1 exit /b 1\ncl /nologo /O2 /std:c++17 /EHsc /LD "{cpp}" /Fe:"{out}"\n',
            encoding="utf-8",
            newline="\n",
        )
        subprocess.run(
            ["cmd", "/d", "/c", str(cmd)], cwd=BUILD, capture_output=True, check=True
        )
    else:
        subprocess.run(
            ["c++", "-O3", "-std=c++17", "-fPIC", "-shared", str(cpp), "-o", str(out)],
            check=True,
        )
    return out


class PublicStockBackend:
    def __init__(self, device):
        self.device = torch.device(device)
        self.gpu = None
        if self.device.type == "cpu":
            self.lib = C.CDLL(str(build()))
            for name, signature in {
                "kp_init": [C.c_void_p] * 2 + [C.c_int] * 4,
                "kp_step": [C.c_void_p] * 3 + [C.c_int],
                "kp_solved": [C.c_void_p] * 2 + [C.c_int],
                "kp_reset": [C.c_void_p, C.c_int],
                "kp_observe": [C.c_void_p] * 3 + [C.c_int],
            }.items():
                fn = getattr(self.lib, name)
                fn.argtypes, fn.restype = signature, None
            return
        self.gpu = gpu = CUDA(self.device)
        program = C.c_void_p()
        gpu.check(
            gpu.nvrtc.nvrtcCreateProgram(
                C.byref(program),
                source(True).encode(),
                b"publicstock.cu",
                0,
                None,
                None,
            ),
            "variant program",
        )
        major, minor = torch.cuda.get_device_capability(gpu.device)
        options = (C.c_char_p * 3)(
            f"--gpu-architecture=compute_{major}{minor}".encode(),
            b"--std=c++17",
            b"--fmad=false",
        )
        try:
            error = gpu.nvrtc.nvrtcCompileProgram(program, 3, options)
            if error:
                size = C.c_size_t()
                gpu.nvrtc.nvrtcGetProgramLogSize(program, C.byref(size))
                log = C.create_string_buffer(size.value)
                gpu.nvrtc.nvrtcGetProgramLog(program, log)
                raise RuntimeError(log.value.decode())
            size = C.c_size_t()
            gpu.check(
                gpu.nvrtc.nvrtcGetPTXSize(program, C.byref(size)), "variant PTX size"
            )
            ptx = C.create_string_buffer(size.value)
            gpu.check(gpu.nvrtc.nvrtcGetPTX(program, ptx), "variant PTX")
        finally:
            gpu.nvrtc.nvrtcDestroyProgram(C.byref(program))
        self.module = C.c_void_p()
        gpu.check(
            gpu.driver.cuModuleLoadData(C.byref(self.module), ptx), "variant load"
        )
        for name in ("kp_init", "kp_step", "kp_solved", "kp_observe", "kp_reset"):
            fn = C.c_void_p()
            gpu.check(
                gpu.driver.cuModuleGetFunction(C.byref(fn), self.module, name.encode()),
                name,
            )
            gpu.functions[name] = fn

    def call(self, name, n, *args):
        if self.gpu is not None:
            self.gpu.call(name, n, *args)
        else:
            getattr(self.lib, name)(
                *[
                    C.c_void_p(x.data_ptr()) if isinstance(x, torch.Tensor) else x
                    for x in args
                ]
            )


class PublicStockBatch:
    def __init__(
        self, seeds, variant="preview0_choice0_stakes0", device="cpu", backend=None
    ):
        self.device = torch.device(device)
        self.variant = VARIANTS[variant]
        self.count = len(seeds)
        self.backend = PublicStockBackend(device) if backend is None else backend
        self.state = torch.empty(
            (self.count, STATE_SIZE), dtype=torch.int32, device=device
        )
        seeds = torch.as_tensor(seeds, dtype=torch.int32, device=device)
        self.backend.call(
            "kp_init",
            self.count,
            self.state,
            seeds,
            self.count,
            self.variant.flags,
            self.variant.rounds,
            int(self.variant.bankrupt),
        )

    def step(self, actions):
        actions = torch.as_tensor(
            actions, dtype=torch.int32, device=self.device
        ).contiguous()
        if actions.shape != (self.count,):
            raise ValueError("One action per environment required")
        result = torch.empty((self.count, 4), dtype=torch.int32, device=self.device)
        self.backend.call(
            "kp_step", self.count, self.state, actions, result, self.count
        )
        if (result[:, 0] == 99).any():
            raise ValueError("Illegal variant action")
        return result

    def solved_actions(self):
        actions = torch.empty(self.count, dtype=torch.int32, device=self.device)
        self.backend.call("kp_solved", self.count, self.state, actions, self.count)
        return actions

    def observe(self):
        obs = torch.empty((self.count, 660), device=self.device)
        legal = torch.empty((self.count, 50), dtype=torch.bool, device=self.device)
        self.backend.call("kp_observe", self.count, self.state, obs, legal, self.count)
        return obs, legal

    def reset_done(self):
        self.backend.call("kp_reset", self.count, self.state, self.count)

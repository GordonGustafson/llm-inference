from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="causal_multihead_self_attention",
      ext_modules=[
          cpp_extension.CUDAExtension(
              "causal_multihead_self_attention",
              ["causal_multihead_self_attention.cu"],
              extra_compile_args={'nvcc': ['-lineinfo', '--use_fast_math']},
              py_limited_api=False)],  # I couldn't get py_limited_api=True to work.
      cmdclass={'build_ext': cpp_extension.BuildExtension},
)

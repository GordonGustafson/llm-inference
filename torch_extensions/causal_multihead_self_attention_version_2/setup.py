from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="causal_multihead_self_attention_version_2",
      ext_modules=[
          cpp_extension.CUDAExtension(
              "causal_multihead_self_attention_version_2",
              ["causal_multihead_self_attention.cu"],
              py_limited_api=False)],  # I couldn't get py_limited_api=True to work.
      cmdclass={'build_ext': cpp_extension.BuildExtension},
)

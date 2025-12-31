import cffi

from IsoSpecPy.isoFFI import isoFFI

isoFFI.ffi.dlopen(str(isoFFI.libpath), isoFFI.ffi.RTLD_GLOBAL | isoFFI.ffi.RTLD_NOLOAD)

from .massimo_cpp_ext import Massimize
from .massimo_cpp_ext import ProblematicInput

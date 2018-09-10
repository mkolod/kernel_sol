import torch
import torch.cuda.nvtx as nvtx
import inspect
from inspect import currentframe, getargvalues, getfullargspec, getmembers, isfunction
import types
import re

class NvtxPatcher:
    
    registry = set()
    nvtx_handle = nvtx._libnvToolsExt()
    
    @staticmethod
    def nvtx_monkey_patch(func):
        def wrapper(*args, **kwargs):
            frame = currentframe()
            v = getargvalues(frame)
            argspec = getfullargspec(func)
            formal_arg_names = argspec.args
            s = "{'op':%s," % v.locals["func"].__name__
            for idx, val in enumerate(v.locals["args"]):
                name = formal_arg_names[idx]
                if isinstance(val, torch.Tensor):
                    name += "_shape"
                    val = tuple(val.size())
                s += "'%s':%s," % (name, str(val))
            num_def=len(argspec.defaults)
            defaults = dict(zip(argspec.args[-num_def:], argspec.defaults))
            overrides={k:str(v) for k, v in v.locals["kwargs"].items()}
            defaults.update(overrides)
            s += "%s}" % str(defaults).strip("{}")
            nvtx.range_push(s)
            result = func(*args, **kwargs)
            nvtx.range_pop()
            return result
        return wrapper
    
    @classmethod
    def register_module(cls, module, regex_filt_lst=None, log=True):
        if not isinstance(regex_filt_lst, list) and regex_filt_lst is not None:
            regex_filt_lst = list(regex_filt_lst)
        if isinstance(module, str):
            module = eval(module)
        name_list = dir(module)
        mod_funcs = [_a for _a in name_list if
                     (isinstance(getattr(module, _a), types.FunctionType) or
                      isinstance(getattr(module, _a), types.BuiltinFunctionType) or
                     isinstance(getattr(module, _a), types.BuiltinMethodType))]
        
        match_any = lambda txt:  any((map(lambda x: re.match(r"%s" % x, txt), regex_filt_lst)))
        if regex_filt_lst is not None:
            function_list = [_x for _x in mod_funcs if match_any(_x)]
        else: 
            function_list = [_x for _x in mod_funcs]
            
        for name in function_list:
            if name in cls.registry:
                continue
            fqn = "{}.{}".format(module.__name__, name)
            temp = eval(fqn)
            patched = NvtxPatcher.nvtx_monkey_patch(temp)
            cls.registry.add(fqn)
            exec("{}=patched".format(fqn))
            
        print("{}\n{}\n".format("Functions registered for NVTX range annotation:", function_list))
        
np = NvtxPatcher.register_module(torch.nn.functional,
                                 ["conv[1-3]?(d|(\_transpose[1-3]d))",
                                 "(un)?fold",
                                  "(avg|max)_pool",
                                 "max_unpool[1-3]d",
                                 "lp_pool[1-3]d",
                                 "adaptive_(avg|max)_pool[1-3]d",
                                 "threshold",
                                 "(leaky_)?[p-s]?r?elu_?6?",
                                 "(hard)?tanh",
                                 "glu",
                                 "(log)?sigmoid",
                                 "(hard|soft|tanh)shrink",
                                 "soft(sign|plus|min)",
                                 "(gumbel_|log_)?softmax",
                                 "(batch|layer|instance|local_response)_norm",
                                 "normalize",
                                 "(bi)?linear",
                                 "(alpha_)?dropout([2-3]d)?",
                                 "embedding(_bag)?",
                                 "pairwise_distance",
                                 "cosine_similarity",
                                 "(binary_)?cross_entropy",
                                 "(poisson_)?nll_loss",
                                 "(cosine|hinge)_embedding_loss",
                                 "kl_div",
                                 "((smooth_)?l1|mse)_loss",
                                 "(multilabel|multi)?_margin_(soft_?)(ranking)?_loss",
                                 "(soft|triplet)_margin_loss",
                                 "pad",
                                 "pixel_shuffle",
                                 "interpolate",
                                 "upsample_?(bilinear|nearest)?",
                                 "(affine_)?grid(_sample)?"])
            
with torch.autograd.profiler.emit_nvtx():

    foo = torch.randn(1, 3, 5, 5).cuda()
    bar = torch.randn(4, 3, 3, 3).cuda()
    result = torch.nn.functional.conv2d(foo, bar)
    print(result) 

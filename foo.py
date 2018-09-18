import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda.nvtx as nvtx
import torch.cuda.profiler as profiler

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
            s = "{'op':'%s'," % v.locals["func"].__name__
            for idx, val in enumerate(v.locals["args"]):
                name = "" + formal_arg_names[idx]
                if isinstance(val, torch.Tensor):
                    name += "_tensor"
                    value = {'shape': tuple(val.size()), 'type': str(val.dtype).split(".")[-1]}
                    val = value
                 #   name += "'"
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
    def list_non_builtins(cls, module, regex_filt_lst=None, log=True):
        if not isinstance(regex_filt_lst, list) and regex_filt_lst is not None:
            regex_filt_lst = list(regex_filt_lst)
        if isinstance(module, str):
            module = eval(module)
        name_list = dir(module)
        builtin_funcs_methods = [_a for _a in name_list if
                                 (isinstance(getattr(module, _a), types.BuiltinFunctionType) or
                                  isinstance(getattr(module, _a), types.BuiltinMethodType))]
        match_any = lambda txt:  any((map(lambda x: re.match(r"%s" % x, txt), regex_filt_lst)))
        if regex_filt_lst is not None:
            function_list = [_x for _x in builtin_funcs_methods if match_any(_x)]
        else: 
            function_list = [_x for _x in builtin_funcs_methods]
        return function_list 
                                 
    @classmethod
    def register_non_builtins(cls, module, regex_filt_lst=None, log=True):
        if not isinstance(regex_filt_lst, list) and regex_filt_lst is not None:
            regex_filt_lst = list(regex_filt_lst)
        if isinstance(module, str):
            module = eval(module)
        name_list = dir(module)
        non_builtin_funcs = [_a for _a in name_list if
                     isinstance(getattr(module, _a), types.FunctionType)]
        
        match_any = lambda txt:  any((map(lambda x: re.match(r"%s" % x, txt), regex_filt_lst)))
        if regex_filt_lst is not None:
            function_list = [_x for _x in non_builtin_funcs if match_any(_x)]
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
            
    @classmethod        
    # convNd is a built-in, so can't be registered using the non-builtin approach above
    def patch_conv(cls, dim_count, module=torch.nn.functional):
        fun_name = "{}.conv{}d".format(module.__name__, str(dim_count))
        # Function already patched
        if fun_name in cls.registry:
            return
        temp = eval(fun_name)
        def decorator(fun):
            def wrapper(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):

                input_dict = {'shape': tuple(input.size()), 'type': str(input.dtype).split(".")[-1]}
                weight_dict = {'shape': tuple(weight.size()), 'type': str(weight.dtype).split(".")[-1]}

                # Interpolate numbers as strings because some can be one-elem tuples as well
                nvtx_str = "{'op':'conv%sd', 'input_tensor':%s, 'weight_tensor':%s, 'stride':%s, 'padding':%s, 'dilation':%s, 'groups':%s}" % (dim_count, str(input_dict), str(weight_dict), str(stride), str(padding), str(dilation), str(groups))
                nvtx.range_push(nvtx_str)
                op = fun(input, weight, bias, stride, padding, dilation, groups)
                nvtx.range_pop()
                return op
            return wrapper
        patched = decorator(temp)
        exec("{}=patched".format(fun_name))
        return patched
 
    @classmethod        
    # convNd is a built-in, so can't be registered using the non-builtin approach above
    def patch_conv_transpose(cls, dim_count, module=torch.nn.functional):
        fun_name = "{}.conv_transpose{}d".format(module.__name__, str(dim_count))
        # Function already patched
        if fun_name in cls.registry:
            return
        temp = eval(fun_name)
        def decorator(fun):
            def wrapper(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):

                input_dict = {'shape': tuple(input.size()), 'type': str(input.dtype).split(".")[-1]}
                weight_dict = {'shape': tuple(weight.size()), 'type': str(weight.dtype).split(".")[-1]}
                # Interpolate numbers as strings because some can be one-elem tuples as well
                nvtx_str = "{'op':'conv_transpose%sd', 'input_tensor':%s, 'weight_tensor':%s, 'stride':%s, 'padding':%s, 'output_padding':%s, 'groups':%s, 'dilation':%s}" % (dim_count, str(input_dict), str(weight_dict), str(stride), str(padding), str(output_padding), str(groups), str(dilation))
                nvtx.range_push(nvtx_str)
                op = fun(input, weight, bias, stride, padding, dilation, groups)
                nvtx.range_pop()
                return op
            return wrapper
        patched = decorator(temp)
        exec("{}=patched".format(fun_name))
        return patched
             
    @classmethod
    def print_registered_functions(cls):
              print("Functions registered for NVTX range annotation:\n{}\n".format(str(cls.registry)))

    
patterns = ["conv[1-3]?(d|(\_transpose[1-3]d))",
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
     "(affine_)?grid(_sample)?"]

NvtxPatcher.register_non_builtins(
    torch.nn.functional, patterns)
                    
for i in range(1, 4):
    NvtxPatcher.patch_conv(i)               
    NvtxPatcher.patch_conv_transpose(i)

print("built-ins (manual monkey-patching required):")
print(NvtxPatcher.list_non_builtins(torch.nn.functional, patterns))
                    
NvtxPatcher.print_registered_functions()


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


with torch.autograd.profiler.emit_nvtx():

  net = LeNet5().cuda()

  bs = 512
  input = torch.randn(bs, 1, 32, 32).cuda()
  out = net(input)

  target = torch.randn(bs, 10).cuda()  # a dummy target, for example
  target = target.view(bs, -1)  # make it the same shape as output
  criterion = nn.MSELoss()

  # create your optimizer
  optimizer = optim.SGD(net.parameters(), lr=0.01)

  # in your training loop:
  optimizer.zero_grad()   # zero the gradient buffers

  profiler.start()
  output = net(input)
  loss = criterion(output, target)
#  loss.backward()
#  optimizer.step()    # Does the update
  profiler.stop()

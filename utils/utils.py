from typing import Callable, Dict, List, Optional, Set
def show_info(x,tag, max_depth=2, n=3,):
    """
    x: 要查看的对象
    max_depth: 最多展开多少层（0 表示只显示当前对象，不展开内部）
    n: 每层最多展示多少个元素
    """
    print("####show info of:",tag)
    import numpy as np
    try:
        import torch
        has_torch = True
    except Exception:
        has_torch = False

    seen_ids = set()  # 防止循环引用

    def _brief_scalar(v):
        """非容器类型的简要描述"""
        t = type(v).__name__
        # torch.Tensor
        if has_torch and isinstance(v, torch.Tensor):
            try:
                return f"<torch.Tensor shape={tuple(v.shape)} dtype={v.dtype} numel={v.numel()}>"
            except Exception:
                return "<torch.Tensor>"
        # numpy.ndarray
        if isinstance(v, np.ndarray):
            return f"<np.ndarray shape={v.shape} dtype={v.dtype} size={v.size}>"
        # 其他
        try:
            r = repr(v)
        except Exception:
            return f"<{t}>"
        if len(r) > 80:
            r = r[:77] + "..."
        return f"<{t} {r}>"

    def _is_container(v):
        if isinstance(v, (dict, list, tuple, set)):
            return True
        if has_torch and isinstance(v, torch.Tensor):
            return True
        if isinstance(v, np.ndarray):
            return True
        return False

    def _print_tensor(x, indent, depth):
        prefix = " " * indent
        try:
            shape = tuple(x.shape)
            numel = x.numel()
            dtype = x.dtype
            print(f"{prefix}torch.Tensor(shape={shape}, dtype={dtype}, numel={numel})")
            flat = x.flatten()
            m = min(n, numel)
            if m > 0:
                print(f"{prefix}  head: {flat[:m]}")
        except Exception:
            print(f"{prefix}<torch.Tensor>")

    def _print_ndarray(x, indent, depth):
        prefix = " " * indent
        try:
            shape = x.shape
            size = x.size
            dtype = x.dtype
            print(f"{prefix}np.ndarray(shape={shape}, dtype={dtype}, size={size})")
            flat = x.ravel()
            m = min(n, size)
            if m > 0:
                print(f"{prefix}  head: {flat[:m]}")
        except Exception:
            print(f"{prefix}<np.ndarray>")

    def _print_any(obj, depth=0, indent=0, key_prefix=""):
        """核心递归打印函数"""
        prefix = " " * indent
        obj_id = id(obj)
        if obj_id in seen_ids:
            print(f"{prefix}{key_prefix}<CyclicRef {type(obj).__name__}>")
            return
        if _is_container(obj):
            seen_ids.add(obj_id)

        tname = type(obj).__name__

        # torch.Tensor
        if has_torch and isinstance(obj, torch.Tensor):
            print(f"{prefix}{key_prefix}", end="")
            _print_tensor(obj, 0, depth)
            return

        # numpy.ndarray
        if isinstance(obj, np.ndarray):
            print(f"{prefix}{key_prefix}", end="")
            _print_ndarray(obj, 0, depth)
            return

        # dict
        if isinstance(obj, dict):
            print(f"{prefix}{key_prefix}dict(len={len(obj)})")
            if depth >= max_depth:
                return
            for i, (k, v) in enumerate(obj.items()):
                if i >= n:
                    print(f"{prefix}  ... ({len(obj) - n} more)")
                    break
                kp = f"[key={repr(k)[:40]}] "
                _print_any(v, depth + 1, indent + 2, kp)
            return

        # list / tuple / set
        if isinstance(obj, (list, tuple, set)):
            # 统一转成 list 方便切片
            xs = list(obj)
            print(f"{prefix}{key_prefix}{tname}(len={len(xs)})")
            if depth >= max_depth:
                return
            for i, v in enumerate(xs[:n]):
                kp = f"[{i}] "
                _print_any(v, depth + 1, indent + 2, kp)
            if len(xs) > n:
                print(f"{prefix}  ... ({len(xs) - n} more)")
            return

        # 其他标量 / 非容器对象
        print(f"{prefix}{key_prefix}{_brief_scalar(obj)}")

    # 顶层：打印总类型，然后展开
    print(f"root type: {type(x)}")
    _print_any(x, depth=0, indent=0)
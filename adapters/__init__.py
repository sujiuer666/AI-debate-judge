"""适配器包：默认**不启用**。

设计约束（硬约束 3）
--------------------
本包内两个适配器都不会自动运行；是否调用由 ``app.py`` 的
``st.session_state["enable_chatlog"]`` / ``st.session_state["enable_qq_upload"]``
决定，且必须在用户显式勾选并填入必要配置后才触发。

导入本包本身不会发起任何网络请求、不读文件、不读环境变量。
"""

from __future__ import annotations

from adapters.qq_export_adapter import load_qq_export
from adapters.chatlog_adapter import ChatlogQuery, build_params, fetch_chatlog

__all__ = ["ChatlogQuery", "build_params", "fetch_chatlog", "load_qq_export"]

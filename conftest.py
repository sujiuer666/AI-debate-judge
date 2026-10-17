"""pytest 引导文件。

作用：把项目根目录加入 ``sys.path``，使 ``from parser import ...`` /
``from adapters... import ...`` 在测试中可用（与运行 ``streamlit run app.py``
时的工作目录假设一致：一律从 ``debate_judge/`` 根目录执行）。

不使用 ``pyproject.toml`` 的 ``pythonpath`` 配置，是为了让本项目保持
「零打包配置、拷走即跑」的 MVP 形态。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

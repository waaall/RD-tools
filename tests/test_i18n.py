from __future__ import annotations

import unittest
from unittest.mock import patch

from core.i18n import LANGUAGE_EN, LANGUAGE_SYSTEM, LANGUAGE_ZH, resolve_locale


class I18nLocaleTests(unittest.TestCase):
    def test_system_locale_uses_first_supported_ui_language(self):
        """系统语言列表里英文优先时,后续中文备用项不能抢占。"""
        with (
            patch("core.i18n.sys.platform", "darwin"),
            patch("core.i18n._macos_ui_language_tags", return_value=["en-US", "zh-Hans-CN"]),
        ):
            self.assertEqual(resolve_locale(LANGUAGE_SYSTEM), LANGUAGE_EN)

    def test_system_locale_skips_unsupported_languages_in_order(self):
        """跳过暂不支持的语言后,继续按系统优先级匹配下一个受支持语言。"""
        with (
            patch("core.i18n.sys.platform", "darwin"),
            patch("core.i18n._macos_ui_language_tags", return_value=["fr-FR", "zh-Hans-CN", "en-US"]),
        ):
            self.assertEqual(resolve_locale(LANGUAGE_SYSTEM), LANGUAGE_ZH)

    def test_posix_language_tag_is_normalized(self):
        """开发态 shell 环境传入 POSIX locale 时,仍能解析主语言码。"""
        with (
            patch("core.i18n.sys.platform", "darwin"),
            patch("core.i18n._macos_ui_language_tags", return_value=["zh_CN.UTF-8"]),
        ):
            self.assertEqual(resolve_locale(LANGUAGE_SYSTEM), LANGUAGE_ZH)


if __name__ == "__main__":
    unittest.main()

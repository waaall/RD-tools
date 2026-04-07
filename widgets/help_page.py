from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import QApplication, QStackedWidget, QTextBrowser, QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, CaptionLabel, CardWidget, SegmentedWidget, TitleLabel

from core.resource_paths import resolve_resource_path


class HelpWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.doc_dir = resolve_resource_path('configs')

        self.setObjectName('AppPage')
        self._build_ui()
        self.show_user_manual()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(16)

        eyebrow = CaptionLabel('DOCUMENTATION', self)
        eyebrow.setObjectName('PageEyebrow')
        main_layout.addWidget(eyebrow)

        title = TitleLabel('帮助', self)
        title.setObjectName('PageTitle')
        main_layout.addWidget(title)

        description = BodyLabel('继续沿用 Markdown 文档源，只把阅读和切换体验统一到 Fluent 界面里。', self)
        description.setObjectName('PageDescription')
        description.setWordWrap(True)
        main_layout.addWidget(description)

        self.segmented = SegmentedWidget(self)
        self.segmented.addItem('user', '用户文档', self.show_user_manual)
        self.segmented.addItem('dev', '开发文档', self.show_develop_manual)
        self.segmented.setCurrentItem('user')
        main_layout.addWidget(self.segmented, 0)

        self.doc_card = CardWidget(self)
        self.doc_card.setObjectName('SurfaceCard')
        card_layout = QVBoxLayout(self.doc_card)
        card_layout.setContentsMargins(16, 16, 16, 16)
        card_layout.setSpacing(12)

        hint = CaptionLabel('文档中的外部链接会继续使用系统浏览器打开。', self.doc_card)
        hint.setObjectName('DocHint')
        hint.setWordWrap(True)
        card_layout.addWidget(hint)

        self.docs_stack = QStackedWidget(self.doc_card)
        self.user_browser = self._create_browser(self.doc_card)
        self.dev_browser = self._create_browser(self.doc_card)
        self.docs_stack.addWidget(self.user_browser)
        self.docs_stack.addWidget(self.dev_browser)
        card_layout.addWidget(self.docs_stack, stretch=1)

        main_layout.addWidget(self.doc_card, stretch=1)

    def _create_browser(self, parent):
        browser = QTextBrowser(parent)
        browser.setObjectName('HelpBrowser')
        browser.setOpenExternalLinks(True)
        return browser

    def show_user_manual(self):
        self.segmented.setCurrentItem('user')
        self.docs_stack.setCurrentWidget(self.user_browser)
        self._display_markdown(os.path.join(self.doc_dir, 'user_manual.md'), self.user_browser)

    def show_develop_manual(self):
        self.segmented.setCurrentItem('dev')
        self.docs_stack.setCurrentWidget(self.dev_browser)
        self._display_markdown(os.path.join(self.doc_dir, 'develop_manual.md'), self.dev_browser)

    def _display_markdown(self, file_path: str, browser: QTextBrowser):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                browser.setMarkdown(file.read())
        except FileNotFoundError:
            browser.setMarkdown(f'Error: File {file_path} not found.')
        except Exception as exc:
            browser.setMarkdown(f'Error reading file {file_path}: {exc}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    trial = HelpWindow()
    trial.show()
    sys.exit(app.exec())

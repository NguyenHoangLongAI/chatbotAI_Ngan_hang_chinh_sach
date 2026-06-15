# RAG_Core/utils/url_formatter.py - NEW UTILITY

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class URLFormatter:
    """
    Utility để format URLs vào câu trả lời
    """

    @staticmethod
    def format_footer_simple(references: List[Dict[str, Any]]) -> str:
        """
        Format đơn giản - Chỉ URLs

        Output:
        📚 Tài liệu tham khảo:
        • https://ngrok.../file1.pdf
        • https://ngrok.../file2.pdf
        """
        refs_with_urls = [ref for ref in references if ref.get('url')]

        if not refs_with_urls:
            return ""

        lines = ["\n\n📚 Tài liệu tham khảo:"]
        for ref in refs_with_urls[:5]:
            lines.append(f"• {ref['url']}")

        return "\n".join(lines)

    @staticmethod
    def format_footer_detailed(references: List[Dict[str, Any]]) -> str:
        """
        Format chi tiết - Filename + URLs

        Output:
        📚 Tài liệu tham khảo:
        1. Thông tư 01/2022/TT-BTTTT
           📎 https://ngrok.../01_2022_TT-BTTTT.pdf
        """
        refs_with_urls = [ref for ref in references if ref.get('url')]

        if not refs_with_urls:
            return ""

        lines = ["\n\n📚 Tài liệu tham khảo:"]

        for i, ref in enumerate(refs_with_urls[:5], 1):
            filename = ref.get('filename', ref.get('document_id', 'Unknown'))
            display_name = filename.rsplit('.', 1)[0] if '.' in filename else filename

            lines.append(f"{i}. {display_name}")
            lines.append(f"   📎 {ref['url']}")

        return "\n".join(lines)

    @staticmethod
    def format_footer_markdown(references: List[Dict[str, Any]]) -> str:
        """
        Format Markdown style - Clickable links

        Output:
        📚 Tài liệu tham khảo:
        • [Thông tư 01/2022/TT-BTTTT](https://ngrok.../file.pdf)
        • [Quyết định 02/2023/QĐ-TTg](https://ngrok.../file2.pdf)
        """
        refs_with_urls = [ref for ref in references if ref.get('url')]

        if not refs_with_urls:
            return ""

        lines = ["\n\n📚 Tài liệu tham khảo:"]

        for ref in refs_with_urls[:5]:
            filename = ref.get('filename', ref.get('document_id', 'Unknown'))
            display_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
            url = ref['url']

            lines.append(f"• [{display_name}]({url})")

        return "\n".join(lines)

    @staticmethod
    def format_footer_html(references: List[Dict[str, Any]]) -> str:
        """
        Format HTML style - For web display

        Output:
        <div class="references">
        <h4>📚 Tài liệu tham khảo:</h4>
        <ol>
        <li><a href="...">Thông tư 01/2022</a></li>
        </ol>
        </div>
        """
        refs_with_urls = [ref for ref in references if ref.get('url')]

        if not refs_with_urls:
            return ""

        lines = [
            '\n\n<div class="references">',
            '<h4>📚 Tài liệu tham khảo:</h4>',
            '<ol>'
        ]

        for ref in refs_with_urls[:5]:
            filename = ref.get('filename', ref.get('document_id', 'Unknown'))
            display_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
            url = ref['url']
            file_type = ref.get('file_type', '').upper().replace('.', '')

            lines.append(
                f'<li><a href="{url}" target="_blank">{display_name}</a> '
                f'<span class="file-type">{file_type}</span></li>'
            )

        lines.extend(['</ol>', '</div>'])

        return '\n'.join(lines)

    @staticmethod
    def format_inline_citations(
            answer: str,
            references: List[Dict[str, Any]]
    ) -> str:
        """
        Format inline citations - URLs trong câu trả lời

        Example:
        Input: "Theo Thông tư 01/2022..."
        Output: "Theo Thông tư 01/2022 [1]..."

        Footer:
        [1] https://ngrok.../01_2022.pdf
        """
        refs_with_urls = [ref for ref in references if ref.get('url')]

        if not refs_with_urls:
            return answer

        # Build citation footer
        citation_lines = ["\n\n---"]
        for i, ref in enumerate(refs_with_urls[:5], 1):
            filename = ref.get('filename', '')
            url = ref['url']
            citation_lines.append(f"[{i}] {url}")

        # Try to add inline citations
        for i, ref in enumerate(refs_with_urls[:5], 1):
            doc_id = ref.get('document_id', '').replace('_', ' ')
            # Simple pattern matching - can be improved
            if doc_id.lower() in answer.lower():
                answer = answer.replace(doc_id, f"{doc_id} [{i}]", 1)

        return answer + '\n'.join(citation_lines)


# Convenience functions
def append_urls_to_answer(
        answer: str,
        references: List[Dict[str, Any]],
        style: str = "detailed"
) -> str:
    """
    Append URLs to answer with specified style

    Args:
        answer: Original answer text
        references: List of references with URLs
        style: "simple" | "detailed" | "markdown" | "html" | "citations"

    Returns:
        Answer with formatted URLs appended
    """
    formatter = URLFormatter()

    if style == "simple":
        footer = formatter.format_footer_simple(references)
    elif style == "markdown":
        footer = formatter.format_footer_markdown(references)
    elif style == "html":
        footer = formatter.format_footer_html(references)
    elif style == "citations":
        return formatter.format_inline_citations(answer, references)
    else:  # detailed (default)
        footer = formatter.format_footer_detailed(references)

    return answer + footer if footer else answer


# Example usage in agent
"""
from utils.url_formatter import append_urls_to_answer

# In generator agent:
answer = llm_model.invoke(prompt)
answer_with_urls = append_urls_to_answer(
    answer, 
    references, 
    style="detailed"  # or "simple", "markdown", "html"
)
"""
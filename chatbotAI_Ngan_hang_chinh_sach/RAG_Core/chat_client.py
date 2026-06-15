# RAG_Core/personalized_chat_client.py

# !/usr/bin/env python3
"""
Personalized Chat Client - Test personalized API
Usage: python personalized_chat_client.py
"""

import requests
import json
import sys


class PersonalizedChatClient:
    def __init__(self, base_url: str = "http://localhost:8502"):
        self.base_url = base_url
        self.session = requests.Session()

    def check_health(self):
        """Kiểm tra tình trạng API"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"🟢 API Status: {health_data['status']}")
                print(f"📊 Message: {health_data['message']}")
                print(f"🎭 Personalization: {'Enabled' if health_data.get('personalization_enabled') else 'Disabled'}")
                return True
            else:
                print(f"🔴 API Error: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            return False

    def send_personalized_message_streaming(
            self,
            question: str,
            history: list = None,
            name: str = "",
            introduction: str = ""
    ):
        """Gửi câu hỏi với personalization (streaming mode)"""
        try:
            payload = {
                "question": question,
                "history": history or [],
                "stream": True,
                "name": name,
                "introduction": introduction
            }

            print("\n" + "=" * 70)
            print(f"👤 Khách hàng: {name or 'Không cung cấp'}")
            print(f"💼 Giới thiệu: {introduction or 'Không cung cấp'}")
            print(f"❓ Câu hỏi: {question}")
            print("=" * 70)
            print("\n💬 Trả lời: ", end='', flush=True)

            with self.session.post(
                    f"{self.base_url}/chat",
                    json=payload,
                    stream=True,
                    timeout=60
            ) as response:

                if response.status_code != 200:
                    print(f"\n🔴 Error: {response.status_code}")
                    return

                personalized = False
                for line in response.iter_lines():
                    if not line:
                        continue

                    line = line.decode("utf-8")

                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]

                    try:
                        chunk_data = json.loads(data_str)
                        chunk_type = chunk_data.get("type")

                        if chunk_type == "chunk":
                            content = chunk_data.get("content", "")
                            print(content, end="", flush=True)

                        elif chunk_type == "references":
                            references = chunk_data.get("references", [])
                            if references:
                                print(f"\n\n📚 Tài liệu tham khảo ({len(references)}):")
                                for i, ref in enumerate(references, 1):
                                    print(f"   {i}. {ref.get('type')}: {ref.get('document_id')}")

                        elif chunk_type == "end":
                            personalized = chunk_data.get("personalized", False)
                            status = chunk_data.get("status", "SUCCESS")
                            print(f"\n\n✅ Status: {status}")
                            print(f"🎭 Personalized: {'Yes' if personalized else 'No'}")

                    except json.JSONDecodeError:
                        continue

            print("=" * 70 + "\n")

        except Exception as e:
            print(f"\n❌ Error: {e}")

    def send_personalized_message_non_streaming(
            self,
            question: str,
            history: list = None,
            name: str = "",
            introduction: str = ""
    ):
        """Gửi câu hỏi với personalization (non-streaming mode)"""
        try:
            payload = {
                "question": question,
                "history": history or [],
                "stream": False,
                "name": name,
                "introduction": introduction
            }

            print("\n" + "=" * 70)
            print(f"👤 Khách hàng: {name or 'Không cung cấp'}")
            print(f"💼 Giới thiệu: {introduction or 'Không cung cấp'}")
            print(f"❓ Câu hỏi: {question}")
            print("=" * 70)
            print("\n⏳ Đang xử lý...")

            response = self.session.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()

                print(f"\n💬 Trả lời:\n{result['answer']}")
                print(f"\n🎭 Personalized: {result.get('personalized', False)}")
                print(f"✅ Status: {result.get('status', 'UNKNOWN')}")

                if result.get("references"):
                    print(f"\n📚 Tài liệu tham khảo ({len(result['references'])}):")
                    for i, ref in enumerate(result['references'], 1):
                        print(f"   {i}. {ref['type']}: {ref['document_id']}")

                print("=" * 70 + "\n")
            else:
                print(f"🔴 Error {response.status_code}: {response.text}")

        except Exception as e:
            print(f"❌ Error: {e}")

    def test_scenarios(self):
        """Test các scenarios khác nhau"""
        print("\n🧪 TESTING PERSONALIZATION SCENARIOS")
        print("=" * 70 + "\n")

        # Scenario 1: CEO level
        print("📋 Scenario 1: CEO Level Customer")
        self.send_personalized_message_streaming(
            question="Nền tảng chuyển đổi số cho doanh nghiệp là gì?",
            name="Nguyễn Hoàng Long",
            introduction="Tổng giám đốc công ty công nghệ và truyền thông VTC NetViet",
            history=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Xin chào quý khách, tôi là trợ lý ảo Onetouch."}
            ]
        )

        input("\n⏸️  Press Enter để tiếp tục scenario 2...")

        # Scenario 2: Manager level
        print("\n📋 Scenario 2: Manager Level Customer")
        self.send_personalized_message_streaming(
            question="Làm sao triển khai AI trong doanh nghiệp?",
            name="Trần Văn An",
            introduction="Trưởng phòng IT tại công ty Công nghệ ABC",
            history=[]
        )

        input("\n⏸️  Press Enter để tiếp tục scenario 3...")

        # Scenario 3: No personalization info
        print("\n📋 Scenario 3: No Personalization Info (Generic)")
        self.send_personalized_message_streaming(
            question="Trợ lý ảo là gì?",
            name="",
            introduction="",
            history=[]
        )

        input("\n⏸️  Press Enter để tiếp tục scenario 4...")

        # Scenario 4: Staff level
        print("\n📋 Scenario 4: Staff Level Customer")
        self.send_personalized_message_streaming(
            question="Hướng dẫn sử dụng Excel cơ bản",
            name="Lê Thị Mai",
            introduction="Nhân viên văn phòng tại công ty XYZ",
            history=[]
        )

    def interactive_mode(self):
        """Chế độ chat tương tác"""
        print("🚀 Personalized Chat Client Started!")
        print("-" * 70)

        if not self.check_health():
            print("❌ Không thể kết nối tới API!")
            return

        print("\n💡 Commands:")
        print("  /test     - Run test scenarios")
        print("  /profile  - Set customer profile (name, introduction)")
        print("  /clear    - Clear customer profile")
        print("  /quit     - Thoát")
        print("\n" + "=" * 70)

        # Customer profile
        customer_name = ""
        customer_introduction = ""
        history = []

        while True:
            try:
                question = input(f"\n❓ Câu hỏi: ").strip()

                if not question:
                    continue

                # Commands
                if question == "/quit":
                    print("👋 Tạm biệt!")
                    break

                elif question == "/test":
                    self.test_scenarios()

                elif question == "/profile":
                    customer_name = input("👤 Tên: ").strip()
                    customer_introduction = input("💼 Giới thiệu: ").strip()
                    print(f"✅ Profile updated: {customer_name} - {customer_introduction[:50]}...")

                elif question == "/clear":
                    customer_name = ""
                    customer_introduction = ""
                    history = []
                    print("🗑️  Đã xóa profile và lịch sử")

                else:
                    # Send question
                    self.send_personalized_message_streaming(
                        question=question,
                        history=history,
                        name=customer_name,
                        introduction=customer_introduction
                    )

                    # Update history
                    history.append({"role": "user", "content": question})
                    # Note: Should capture actual answer, but simplified for demo

            except KeyboardInterrupt:
                print("\n👋 Tạm biệt!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("🎭 PERSONALIZED RAG CHATBOT CLIENT")
    print("=" * 70 + "\n")

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test mode
        client = PersonalizedChatClient()
        if client.check_health():
            client.test_scenarios()
    else:
        # Interactive mode
        client = PersonalizedChatClient()
        client.interactive_mode()


if __name__ == "__main__":
    main()
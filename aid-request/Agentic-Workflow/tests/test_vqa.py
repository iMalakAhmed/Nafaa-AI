import sys
import os
import unittest
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.nodes.vqa import vqa_node

class TestVQAModule(unittest.TestCase):

    def setUp(self):
        """إعداد بيئة الاختبار قبل كل حالة."""
        self.sample_image = "data/test_image.jpg"
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.sample_image):
            with open(self.sample_image, "w") as f:
                f.write("dummy content")

    def test_vqa_single_image(self):
        """اختبار الحالة: صورة واحدة مع أسئلة افتراضية."""
        state = {
            "text": "تحقق من الصورة",
            "images": [self.sample_image],
            "evidence": {},
            "reasoning": {}
        }
        result = vqa_node(state)
        self.assertIn("vqa_analysis", result["evidence"])
        print("\n✅ Test VQA Single Image: Passed")

    def test_vqa_multiple_images(self):
        """اختبار الحالة: صور متعددة مع أسئلة محددة."""
        state = {
            "text": "تحليل الأدوية",
            "images": [self.sample_image, self.sample_image],
            "evidence": {},
            "reasoning": {
                "instruction": {
                    "query_or_question": "ما اسم الدواء الظاهر؟"
                }
            }
        }
        result = vqa_node(state)
        self.assertGreaterEqual(len(result["evidence"]["vqa_analysis"]), 1)
        print("✅ Test VQA Multiple Images: Passed")

    def test_vqa_no_images(self):
        """اختبار الحالة: عدم وجود صور (يجب أن يتعامل النظام بمرونة)."""
        state = {
            "text": "لا توجد صور",
            "images": [],
            "evidence": {},
            "reasoning": {}
        }
        # نتوقع ألا ينهار النظام
        try:
            result = vqa_node(state)
            self.assertEqual(result["evidence"]["vqa_analysis"], [])
            print("✅ Test VQA No Images: Passed")
        except Exception as e:
            self.fail(f"vqa_node failed with no images: {e}")

    def test_vqa_invalid_image_path(self):
        """اختبار الحالة: مسار صورة غير موجود."""
        state = {
            "text": "صورة خاطئة",
            "images": ["non_existent.jpg"],
            "evidence": {},
            "reasoning": {}
        }
        result = vqa_node(state)
        self.assertIn("error", str(result["evidence"]))
        print("✅ Test VQA Invalid Image Path: Passed")

if __name__ == '__main__':
    unittest.main()
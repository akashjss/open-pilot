# screens.py
import pyautogui
import os
from PIL import Image

class Screens:
    def save_screenshot_to_file(self, filename: str = "screenshot.jpg", quality: int = 85) -> str:
        """
        Captures the current screen, compresses it, and saves it to a file.

        Args:
            filename (str): The name of the file to save the screenshot.
            quality (int): The quality for the JPEG compression (1-100).

        Returns:
            str: The path to the saved screenshot file.
        """
        try:
            # Capture the screenshot using pyautogui
            screenshot = pyautogui.screenshot()
            temp_png_filename = filename.replace(".jpg", ".png")
            screenshot.save(temp_png_filename)

            # Compress and convert the screenshot to JPEG format
            with Image.open(temp_png_filename) as img:
                img = img.convert("RGB")  # Ensure the image is in RGB format for JPEG
                img.save(filename, "JPEG", quality=quality)

            # Remove the temporary PNG file
            os.remove(temp_png_filename)
            
            print(f"Screenshot saved to {filename}")
            return os.path.abspath(filename)
        except Exception as e:
            print(f"Failed to save screenshot: {e}")
            raise

    def delete_temp_screenshot_files(self, filename: str = "screenshot.jpg") -> None:
        """
        Deletes the specified screenshot file if it exists.

        Args:
            filename (str): The name of the file to delete.
        """
        try:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Deleted screenshot file: {filename}")
            else:
                print(f"No screenshot file found at: {filename}")
        except Exception as e:
            print(f"Failed to delete screenshot file: {e}")
            raise

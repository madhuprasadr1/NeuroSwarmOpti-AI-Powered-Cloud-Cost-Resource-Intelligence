import os
import subprocess
import time

def main():
    html_path = os.path.abspath(os.path.join("outputs", "CloudOpt_AI_Judges_Presentation_Guide.html"))
    pdf_path = os.path.abspath(os.path.join("outputs", "CloudOpt_AI_Judges_Presentation_Guide.pdf"))
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    
    file_url = "file:///" + html_path.replace("\\", "/")
    
    cmd = [
        edge_path,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={pdf_path}",
        "--no-pdf-header-footer",
        file_url
    ]
    
    print(f"Converting {html_path} to PDF...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("Edge process completed with code:", result.returncode)
    
    time.sleep(1)
    if os.path.exists(pdf_path):
        size = os.path.getsize(pdf_path)
        print(f"SUCCESS: Generated PDF at:\n{pdf_path}\nSize: {size:,} bytes")
    else:
        print("ERROR: PDF was not generated.")

if __name__ == "__main__":
    main()

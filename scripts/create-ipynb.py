import subprocess
from dotenv import load_dotenv
from pathlib import Path

live_notebook_dir = Path("docs/live-notebooks")
live_notebook_dir.mkdir(parents=True, exist_ok=True)

for qmd in Path("chapters").glob("*"): 
    
    if not qmd.suffix == ".qmd":
        continue
    
    out_path = live_notebook_dir / f"{qmd.stem}.ipynb"  
        
    # if not out_path.exists() or out_path.stat().st_mtime < qmd.stat().st_mtime:
    subprocess.run(
                ["quarto", "render", str(qmd), "--profile", "live-notebook", "--to", "ipynb", "--output", out_path.name, "--no-execute"], 
                check=True
    )
    
    # subprocess.run(
    #             ["quarto", "render", str(qmd), "--profile", "live-notebook", "--to", "native",  "--no-execute"], 
    #             check=True
    # )
    # else: 
        # print(f"Skipping {qmd}, up to date.")

    
    
    
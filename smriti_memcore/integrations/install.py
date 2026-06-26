import sys
import subprocess

def main():
    print("SMRITI Memory System Installer")
    print("==============================")
    
    if sys.platform in ("linux", "darwin"):
        print("Launching the automated bash installer...")
        try:
            # Fetch and run the installer bash script directly
            cmd = "curl -fsSL https://raw.githubusercontent.com/smriti-memcore/smriti-memcore/main/install_smriti_mcp.sh | bash"
            subprocess.run(cmd, shell=True, check=True)
        except Exception as e:
            print(f"\nError running bash installer: {e}")
            print("Please run it manually in your shell:")
            print("  curl -fsSL https://raw.githubusercontent.com/smriti-memcore/smriti-memcore/main/install_smriti_mcp.sh | bash")
    else:
        print("\nWindows / non-POSIX platform detected.")
        print("Please configure your MCP servers manually using the following parameters:")
        print("\nCommand:")
        print("  python -m smriti_memcore.integrations.mcp_server")
        print("\nEnvironment Variables:")
        print("  SMRITI_STORAGE_PATH: path to your memory directory (e.g., ~/.smriti/global)")
        print("  SMRITI_LLM_MODEL: model name (e.g., gemini-2.5-flash)")
        print("  SMRITI_LLM_API_KEY: your API key (if using cloud model)")
        print("  SMRITI_OBSIDIAN_PATH: path to your Obsidian vault palace folder (optional)")

if __name__ == "__main__":
    main()

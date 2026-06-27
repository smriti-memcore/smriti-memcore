import re
import json
import pathlib

def bump_version():
    # 1. Read pyproject.toml
    toml_path = pathlib.Path('pyproject.toml')
    toml_text = toml_path.read_text()
    
    # Extract and increment version
    match = re.search(r'^version\s*=\s*\"(\d+)\.(\d+)\.(\d+)\"', toml_text, re.MULTILINE)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")
        
    major, minor, patch = match.groups()
    new_version = f"{major}.{minor}.{int(patch) + 1}"
    
    # Write new version to pyproject.toml
    new_toml_text = re.sub(
        r'^version\s*=\s*\"[^\"]+\"',
        f'version = "{new_version}"',
        toml_text,
        flags=re.MULTILINE
    )
    toml_path.write_text(new_toml_text)
    print(f"Bumped pyproject.toml to {new_version}")

    # 2. Update server.json
    server_path = pathlib.Path('server.json')
    if server_path.exists():
        server_data = json.loads(server_path.read_text())
        server_data['version'] = new_version
        if 'packages' in server_data and len(server_data['packages']) > 0:
            server_data['packages'][0]['version'] = new_version
        server_path.write_text(json.dumps(server_data, indent=2) + '\n')
        print(f"Bumped server.json to {new_version}")
        
    return new_version

if __name__ == '__main__':
    bump_version()

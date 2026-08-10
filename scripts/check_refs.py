import re

with open(r'D:\01_Projects\INTERSYMBOLIC-GRC\thesis\references.bib', encoding='utf-8') as f:
    bib = f.read()

with open(r'D:\01_Projects\INTERSYMBOLIC-GRC\thesis\INTERSYMBOLIC-GRC_Thesis.tex', encoding='utf-8') as f:
    tex = f.read()

bib_keys = set(re.findall(r'@\w+\{([^,]+),', bib))

# The tex file uses \cite{key} - on Windows the file has CRLF and the
# backslash might be escaped. Use a simple string search approach.
cited_keys = set()
pos = 0
while True:
    idx = tex.find(r'\cite{', pos)
    if idx == -1:
        break
    end = tex.find('}', idx)
    if end == -1:
        break
    chunk = tex[idx+6:end]
    for k in chunk.split(','):
        cited_keys.add(k.strip())
    pos = end + 1

unused = sorted(bib_keys - cited_keys)
missing = sorted(cited_keys - bib_keys)

print('=== UNUSED IN .bib (defined but never cited) ===')
for k in unused:
    print(' ', k)
print()
print('=== CITED BUT MISSING FROM .bib (undefined keys) ===')
for k in missing:
    print(' ', k)
print()
print(f'Bib total: {len(bib_keys)}, Cited unique: {len(cited_keys)}, Unused: {len(unused)}, Missing: {len(missing)}')
print()
print('Sample cited keys:', sorted(cited_keys)[:10])

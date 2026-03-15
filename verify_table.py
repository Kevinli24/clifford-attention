import clifford as cf

layout, blades = cf.Cl(3)

e1   = blades['e1']
e2   = blades['e2']
e3   = blades['e3']
e12  = blades['e12']
e13  = blades['e13']
e23  = blades['e23']
e123 = blades['e123']
one  = 1.0 * layout.scalar  

basis = [one, e1, e2, e3, e12, e13, e23, e123]
names = ['1','e1','e2','e3','e12','e13','e23','e123']

print("Cl(3,0) Geometric Product Table")
print("Rows = left operand, Cols = right operand")
print(f"{'':>8}", end="")
for n in names:
    print(f"{n:>10}", end="")
print()

for i, (a, na) in enumerate(zip(basis, names)):
    print(f"{na:>8}", end="")
    for j, (b, nb) in enumerate(zip(basis, names)):
        prod = a * b
        s = str(prod).strip()
        s = s.replace('(1.0)', '1').replace('(-1.0)', '-1').replace(' ', '')
        print(f"{s:>10}", end="")
    print()
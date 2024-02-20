s = [0, 0, 24]
v = []
for i in range(0,101):
    v.append(i)
if s[2]==99:
    s[2]=0
    if s[1]==99:
        s[1]=0
        if s[0]==99:
            s[0]=0
        else:
            s[0]=v[s[0]+1]
    else:
        s[1]=v[s[1]+1]
else:
    s[2]=v[s[2]+1]
print(s)

import streamlit as st

st.header("Primitive Root")

def prime(q, g):
    if not q.isnumeric() or not g.isnumeric():
        return
    
    q = int(q)  # Convert q to an integer
    g = int(g)  # Convert g to an integer
    
    x = 0
    
    for i in range(1, q+1):
        if q % i == 0:
            x += 1
    
    if x > 2:
        st.snow()
        st.write(f"{q} is not a prime number!!")
    elif x == 2:
        primitive = False  # Initialize primitive as False
        
        l = []
        isp = []
        
        for i in range(q-1):
            for o in range(q):
                n = i + 1
                m = o + 1
                z = n**m % q
                
                if z not in l:
                    l.append(z)
                    if m <= (q-2):
                        st.write(f"{n}^{m} mod {q} = {z}", end=", ")
                        
                    if m == (q-1):
                        st.write(f"{n}^{m} mod {q} = {z} ==> {n} is a primitive root of {q}", end=", " )
                        primitive = True
                        isp.append(n)
                        
                elif z in l:
                    l.clear()
                    st.write("")
                    break
        
        if g in isp:
            st.snow()
            st.write(f"{g} is a primitive root: {primitive} {isp}")
        else:
            st.snow()
            st.write(f"{g} is NOT a primitive root of {q} - List of Primitive roots: {isp}")

    return
    
q = st.text_input("Prime number:")
g = st.text_input("Primitive:")

if st.button("Submit"):
    if not g:
        st.error("Invalid Input!")
    elif not q:
        st.error("Invalid Input!")
    else:
        prime(q, g)
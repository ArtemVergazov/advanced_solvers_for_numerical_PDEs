from dolfin import *
import mshr
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from subprocess import call
import sys

# Number of mesh points
N=40

# (Convert to SI units at the end)
# Distance between cylinders
a=1.0

# Radius of cylinders
R=0.4

# Wave speed
c=1.0

# Experimental parameters
# Density and speed of sound in cylinders
rhoc=2200
cc=5700
# Density and speed of sound in helium
rhoh=190
ch=200
# Spacing between centre of cylinders
s=6.0E-5
# Height of cylinders 
h=200E-9

# For checking
# From COMSOL, at M point (EM simulation with a PMC boundary)
f1=1.14E8
f2=1.76E8
aC=1.
cC=3.E8
# Convert to dimensionless form
f1pmc=f1*(aC/cC)
f2pmc=f2*(aC/cC)

rt=((cc*rhoc/(ch*rhoh)) - 1.0)/((cc*rhoc/(ch*rhoh)) + 1.0)
print("r={0}".format(rt))

f1=0.5*np.pi*ch/(np.pi*h)
print("Cut-off frequency={0} MHz".format(f1*1E-6))

# The lattice vectors
phuc=np.pi/4.0
a1=a*np.array([np.cos(phuc),np.sin(phuc)])
a2=a*np.array([np.cos(phuc),-np.sin(phuc)])

# Reciprocal space lattice
b1=(2.0*np.pi)*np.array([a2[1],-a2[0]])/(a1[0]*a2[1]-a2[0]*a1[1])
b2=(2.0*np.pi)*np.array([a1[1],-a1[0]])/(a1[1]*a2[0]-a2[1]*a1[0])

# Set up mesh for 2D problem
# vertices of unit cell (in anticlockwise order)
# Note that the mesh is constructed so that the periodic boundary conditions are between the side V[0]->V[1] and V[2]->V[3]
V=[[],[],[],[]]
V[0]=0.5*(-a1+a2)
V[1]=0.5*(a1+a2)
V[2]=0.5*(a1-a2)
V[3]=-0.5*(a1+a2)

# Position of cylinder in unit cell
Xc=np.array([0,0])

# Points to construct circle from (see gmsh documentation on Circle)
C=[[],[],[],[],[]]
C[0]=[0,0]
C[1]=[R,0]
C[2]=[-R,0]
C[3]=[0,-R]
C[4]=[0,R]

# Corners of the unit cell
point_script="\n// Mesh definition\n// Outer boundary\n"
for i in np.arange(4):
    point_script+="Point({0})={{{1},{2},0,gridsize}};\n".format(i+1,V[i][0],V[i][1])

# The circle to cut out of the middle of the unit cell
circle_script="// Inner circle\n"
for i in np.arange(5):
    circle_script+="Point({0})={{{1},{2},0,gridsize}};\n".format(i+5,Xc[0]+C[i][0],Xc[1]+C[i][1])

circle_script+="\nCircle(5) = {9, 5, 6};\nCircle(6) = {6, 5, 8};\nCircle(7) = {8, 5, 7};\nCircle(8) = {7, 5, 9};\n"

# Convert to a gmsh script, run gmsh and then use dolfin-convert to get the mesh into xml format
# Script components sides=1;
param_script="// Inputs\n\ngridsize=1/{0};\nradius={1};\n".format(N,R)
line_script="Line(1) = {1, 2};\nLine(2) = {2, 3};\nLine(3) = {3, 4};\nLine(4) = {4, 1};\n"
line_loop_script="\nLine Loop(9) = {5, 6, 7, 8};\n\nLine Loop(10) = {4, 1, 2, 3};\n"
periodic_bc_script="Periodic Line {1}={3};\nPeriodic Line {2}={4};"
surface_script="\nPlane Surface(6) = {10,9};\n"

# Total script
mesh_script=param_script+point_script+line_script+circle_script+line_loop_script+periodic_bc_script+surface_script

# Run gmsh and dolfin-convert to get the xml mesh file
print("Running Gmsh and dolfin-convert...")
geo_file=open("acoustics_mesh_file.geo",'w')
geo_file.write(mesh_script)
geo_file.close()
# gmsh_err=call(["gmsh", "-2", "acoustics_mesh_file.geo", "-o", "amf.msh"])
gmsh_err=call(["gmsh", "-2", "acoustics_mesh_file.geo", "-format", "msh2", "-o", "amf.msh"])
dolfin_err = False
if gmsh_err:
    print("...something bad happened when the mesh was being generated...")
else:
    dolfin_err=call(["dolfin-convert", "amf.msh", "amf.xml"])
if dolfin_err:
    print("...something bad happened when the mesh was being generated...")
else:
    print("...mesh generated successfully!")
# Load the xml mesh
mesh=Mesh("amf.xml")
print("...and loaded into Fenics!")

# View mesh
plot(mesh,backend="matplotlib")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.title("Mesh of unit cell")
#plt.title("${\\rm Mesh\;of\;unit\;cell\;(indicating\;identified\;sides)}$")
plt.xlabel("x/a",fontsize=18)
plt.ylabel("y/a",fontsize=18)
plt.ylim(0.5*(-a1[1]+a2[1]),0.5*(a1[1]-a2[1]))
plt.xlim(0.5*(-a1[0]-a2[0]),0.5*(a1[0]+a2[0]))
plt.plot([V[0][0],V[1][0]],[V[0][1],V[1][1]],'b-',lw=2)
plt.plot([V[1][0],V[2][0]],[V[1][1],V[2][1]],'r-',lw=2)
plt.plot([V[3][0],V[0][0]],[V[3][1],V[0][1]],'r-',lw=2)
plt.plot([V[2][0],V[3][0]],[V[2][1],V[3][1]],'b-',lw=2)
# Alternative plotting option, before I discovered the matplotlib option...
# HTML(X3DOM().html(mesh))
plt.savefig('fig1.png')
plt.close()

# Sub domain for Periodic boundary condition
Mtol=0.001 # Tolerance when I ask whether the value of a mesh point is x 

# Is the point on the lower left boundary?
def lr(x):
    t1=0.5*(a1[0]*a2[1]-a2[0]*a1[1])/a1[0]
    g=a1[1]/a1[0]
    return bool((x[1]-g*x[0]-t1)**2.0<Mtol**2.0)
# Is the point on the lower right boundary?
def ll(x):
    t1=0.5*(a1[0]*a2[1]-a2[0]*a1[1])/a1[0]
    g=a2[1]/a2[0]
    return bool((x[1]-g*x[0]-t1)**2.0<Mtol**2.0)
# Is the point on the upper right boundary?
def ur(x):
    t1=0.5*(a1[0]*a2[1]-a2[0]*a1[1])/a1[0]
    g=a2[1]/a2[0]
    return bool((x[1]-a1[1]-g*(x[0]-a1[0])-t1)**2.0<Mtol**2.0)

class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((ll(x) or lr(x)) and 
                (not ((near(x[0], -0.5*(a1[0]+a2[0])) and near(x[1], -0.5*(a1[1]+a2[1]))) or 
                        (near(x[0], 0.5*(a1[0]+a2[0])) and near(x[1], 0.5*(a1[1]+a2[1])))) 
                 or (x[0]-Xc[0])**2.0 + (x[1]-Xc[1])**2.0 - R**2.0 < Mtol**2.0) and on_boundary)

    def map(self, x, y):
        # The top-most boundary point is identified with the lower-most one
        if near(x[0], 0.5*(a1[0]-a2[0])) and near(x[1], 0.5*(a1[1]-a2[1])):
            y[0] = x[0] - (a1[0]-a2[0])
            y[1] = x[1] - (a1[1]-a2[1])
        # The upper right boundary is shifted by a_1
        elif ur(x):
            y[0] = x[0] - a1[0]
            y[1] = x[1] - a1[1]
        # Otherwise shift by a_2
        else:   # near(x[1], 1)
            y[0] = x[0] + a2[0]
            y[1] = x[1] + a2[1]

# Solver 
def esol(Kx,Ky):
    K=Constant((Kx,Ky))
    # For complex fields you need to double the Function space...
    Vr = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    Vi = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    Vc = Vr*Vi
    Vz = FunctionSpace(mesh,Vc,constrained_domain=PeriodicBoundary())
    # ...and then define separate real and imaginary parts for the test and trial function
    (u_r, u_i) = TrialFunctions(Vz)
    (v_r, v_i) = TestFunctions(Vz)
    # Define differential equation in weak (integral) form - term by term
    # Term 1
    t1r=inner(grad(u_r), grad(v_r))-inner(grad(u_i), grad(v_i))
    t1i=inner(grad(u_r), grad(v_i))+inner(grad(u_i), grad(v_r))
    # Term 2
    t2r=v_i*inner(K,grad(u_r)) + v_r*inner(K,grad(u_i)) - u_i*inner(K,grad(v_r)) - u_r*inner(K,grad(v_i))
    t2i=-v_r*inner(K,grad(u_r)) + v_i*inner(K,grad(u_i)) + u_r*inner(K,grad(v_r)) - u_i*inner(K,grad(v_i)) 
    # Term 3
    t3r=inner(K,K)*(u_r*v_r-u_i*v_i)
    t3i=inner(K,K)*(u_r*v_i+u_i*v_r)

    # Sum terms and define a useless term
    ar = t1r+t2r+t3r
    ai = t1i+t2i+t3i
    L = Constant(0.0)*(v_r+v_i)*dx

    # Overlap integral between test and trial functions
    m = (u_r*v_r - u_i*v_i + u_r*v_i + u_i*v_r)*dx

    A,_ = assemble_system((ar+ai)*dx, L)#, bc)
    B = assemble(m)

    eigensolver = SLEPcEigenSolver(as_backend_type(A), as_backend_type(B))
    prm = eigensolver.parameters
    info(prm, True)
    eigensolver.parameters['spectrum'] = 'smallest magnitude'

    eigensolver.solve(20)

    return eigensolver

pnts=30

# The K point
KK=KK=0.5*(b1+b2)-0.5*np.dot(b1,b2)*np.array([-b1[1]-b2[1],b1[0]+b2[0]])/(b1[0]*b2[1]-b1[1]*b2[0])

# A sequence of K points, on a route around the Brillouin zone
Ks1=[0.5*(b1+b2)*i for i in np.linspace(0.01,1.0,pnts)]
Ks2=[0.5*(b1+b2)+(KK-0.5*(b1+b2))*i for i in np.linspace(0.01,1.0,pnts)]
Ks3=[KK*(1.0-i) for i in np.linspace(0.0,0.99,pnts)]
Kv=Ks1+Ks2+Ks3

sol=[esol(i[0],i[1]) for i in tqdm.tqdm(Kv)]

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot([np.sqrt(i.get_eigenpair(0)[0])*0.5*ch*1E-6/(s*np.pi) for i in sol])
plt.plot([np.sqrt(i.get_eigenpair(2)[0])*0.5*ch*1E-6/(s*np.pi) for i in sol])
plt.plot([np.sqrt(i.get_eigenpair(4)[0])*0.5*ch*1E-6/(s*np.pi) for i in sol])
plt.plot([np.sqrt(i.get_eigenpair(6)[0])*0.5*ch*1E-6/(s*np.pi) for i in sol])
# Tow sample points taken from an analogous simulation in COMSOL (R=0.4, phiuc=pi/6)
#plt.plot([pnts],[f1pmc*1E-6*ch/s],'bo')
#plt.plot([pnts],[f2pmc*1E-6*ch/s],'bo')
plt.xticks([0,pnts,2*pnts,3*pnts],["$\Gamma$","$M$","$K$","$\Gamma$"],fontsize=18)
plt.ylabel("f (${\\rm \\times 10^{6}\;Hz}$)",fontsize=18)
plt.savefig('fig2.png')
plt.close()

# Plot pressure field
N=4 # Point on K-axis
n=3 # Eigenvalue to plot
Vr = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
Vi = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
Vc = Vr*Vi
Vz = FunctionSpace(mesh,Vc,constrained_domain=PeriodicBoundary())
pressure = Function(Vz)
eig=pressure.vector()
eig[:]=sol[N].get_eigenpair(2*n)[2]

plt.title("Eigenmode {0} for Kx={1:.1f}, Ky={2:.1f}".format(n,Kv[N][0],Kv[N][1]))
plot(pressure[0],backend="matplotlib")
plt.xlabel("$x/a$",fontsize=18)
plt.ylabel("$y/a$",fontsize=18)
plt.savefig('fig3.png')
plt.close()

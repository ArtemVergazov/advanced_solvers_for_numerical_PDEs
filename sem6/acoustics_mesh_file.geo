// Inputs

gridsize=1/40;
radius=0.4;

// Mesh definition
// Outer boundary
Point(1)={0.0,-0.7071067811865475,0,gridsize};
Point(2)={0.7071067811865476,0.0,0,gridsize};
Point(3)={0.0,0.7071067811865475,0,gridsize};
Point(4)={-0.7071067811865476,-0.0,0,gridsize};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
// Inner circle
Point(5)={0,0,0,gridsize};
Point(6)={0.4,0,0,gridsize};
Point(7)={-0.4,0,0,gridsize};
Point(8)={0,-0.4,0,gridsize};
Point(9)={0,0.4,0,gridsize};

Circle(5) = {9, 5, 6};
Circle(6) = {6, 5, 8};
Circle(7) = {8, 5, 7};
Circle(8) = {7, 5, 9};

Line Loop(9) = {5, 6, 7, 8};

Line Loop(10) = {4, 1, 2, 3};
Periodic Line {1}={3};
Periodic Line {2}={4};
Plane Surface(6) = {10,9};

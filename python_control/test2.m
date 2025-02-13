clc;
clear;
close all;

% Define symbolic variables
syms alpha s;

% Define the state-space matrices as symbolic
A = sym([2 1; -1 -1]); % 2x2 matrix
B = sym([0; 1]);       % 2x1 matrix
C = sym([-10 1]);      % 1x2 matrix
D = sym(0);            % No direct feedthrough

% Create the symbolic phi matrix
I = eye(size(A));

% Compute phi matrix symbolically
tl = A;
tr = B;
bl = -inv(C * inv(A) * (expm(A * 0.25) - I) * B) * C * expm(A * 0.25);
br = -eye(1);

% Multiply bl and br by alpha
bl = alpha * bl;
br = alpha * br;

% Construct the modified phi_alpha matrix
phi_alpha = [[tl, tr]; [bl, br]];

% Compute the characteristic polynomial of phi_alpha
char_poly = simplify(det(s * eye(size(phi_alpha)) - phi_alpha));

% Extract terms multiplied by alpha
alpha_terms = collect(char_poly, alpha) - subs(char_poly, alpha, 0);

% Solve for roots of the alpha terms in floating-point form
alpha_roots = double(solve(alpha_terms == 0, s));

% Display the result
disp('Floating-point roots of the alpha terms in the characteristic polynomial:');
disp(alpha_roots);

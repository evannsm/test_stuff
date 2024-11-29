/*
 * This file was autogenerated by TinyMPC on Mon Oct 21 13:08:42 2024
 */

#include <iostream>

#include <tinympc/tiny_api.hpp>
#include <tinympc/tiny_data.hpp>

using namespace Eigen;
IOFormat TinyFmt(4, 0, ", ", "\n", "[", "]");

#ifdef __cplusplus
extern "C" {
#endif

int main()
{
	int exitflag = 1;
	// Double check some data
	std::cout << "rho: " << tiny_solver.cache->rho << std::endl;
	std::cout << "\nmax iters: " << tiny_solver.settings->max_iter << std::endl;
	std::cout << "\nState transition matrix:\n" << tiny_solver.work->Adyn.format(TinyFmt) << std::endl;
	std::cout << "\nInput/control matrix:\n" << tiny_solver.work->Bdyn.format(TinyFmt) << std::endl;

	// Visit https://tinympc.org/ to see how to set the initial condition and update the reference trajectory.

	std::cout << "\nSolving...\n" << std::endl;

	exitflag = tiny_solve(&tiny_solver);

	if (exitflag == 0) printf("Hooray! Solved with no error!\n");
	else printf("Oops! Something went wrong!\n");
	return 0;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

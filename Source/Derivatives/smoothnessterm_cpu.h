#ifndef SMOOTHNESSTERM_H
#define SMOOTHNESSTERM_H

#include <iostream>
#include "../Solver/optflow_base.h"

namespace optflow
{
namespace cpu2d
{
	void update_smoothnessterm_Barron(optflow_type *u, optflow_type *du,  optflow_type *phi, mathtype_solver epsilon_phi_squared, int shape[3], mathtype_solver hx = 1.f, mathtype_solver hy = 1.f);
	void update_smoothnessterm_centralDiff(optflow_type *u, optflow_type *du,  optflow_type *phi, mathtype_solver epsilon_phi_squared, int shape[3],mathtype_solver hx = 1.f, mathtype_solver hy = 1.f);
	void update_smoothnessterm_forwardDiff(optflow_type *u, optflow_type *du,  optflow_type *phi, mathtype_solver epsilon_phi_squared, int shape[3], mathtype_solver hx = 1.f, mathtype_solver hy = 1.f);
}

namespace cpu3d
{
	void update_smoothnessterm_Barron(optflow_type *u, optflow_type *du,  optflow_type *phi, mathtype_solver epsilon_phi_squared, int shape[3], mathtype_solver hx = 1.f, mathtype_solver hy = 1.f, mathtype_solver hz = 1.f);
	void update_smoothnessterm_centralDiff(optflow_type *u, optflow_type *du,  optflow_type *phi, mathtype_solver epsilon_phi_squared, int shape[3], mathtype_solver hx = 1.f, mathtype_solver hy = 1.f, mathtype_solver hz = 1.f);
	void update_smoothnessterm_forwardDiff(optflow_type *u, optflow_type *du,  optflow_type *phi, mathtype_solver epsilon_phi_squared, int shape[3], mathtype_solver hx = 1.f, mathtype_solver hy = 1.f, mathtype_solver hz = 1.f);
}
}
#endif //SMOOTHNESSTERM_H

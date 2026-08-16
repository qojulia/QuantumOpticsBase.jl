const total_started_ns = time_ns()
const import_started_ns = total_started_ns
using QuantumOpticsBase
const import_seconds = (time_ns() - import_started_ns) / 1.0e9

check(condition, message) = condition || error(message)

function fock()
    basis = FockBasis(20)
    alpha = 1.2
    annihilation = destroy(basis)
    state = coherentstate(basis, alpha)
    density = dm(state)
    derivative = -1im * (annihilation + create(basis)) * state

    check(isapprox(expect(annihilation, state), alpha; atol=1e-10), "coherent-state amplitude is incorrect")
    check(isapprox(expect(number(basis), density), abs2(alpha); atol=1e-9), "photon number is incorrect")
    check(isapprox(tr(density), 1; atol=1e-12), "density operator is not normalized")
    check(isfinite(norm(derivative)), "Fock-space operator application returned a non-finite result")
    return real(expect(number(basis), state))
end

function composite()
    spin_basis = SpinBasis(1 // 2)
    raising = sigmap(spin_basis)
    lowering = sigmam(spin_basis)
    atom_hamiltonian = 2 * raising * lowering

    fock_basis = FockBasis(20)
    annihilation = destroy(fock_basis)
    field_hamiltonian = number(fock_basis)
    interaction = annihilation ⊗ raising + create(fock_basis) ⊗ lowering

    basis = fock_basis ⊗ spin_basis
    hamiltonian =
        embed(basis, 1, field_hamiltonian) +
        embed(basis, 2, atom_hamiltonian) +
        interaction
    initial_state = fockstate(fock_basis, 1) ⊗ spindown(spin_basis)
    derivative = hamiltonian * initial_state
    reduced_field = ptrace(dm(initial_state), 2)

    check(QuantumOpticsBase.basis(derivative) == basis, "composite operator changed the state basis")
    check(isapprox(tr(reduced_field), 1; atol=1e-12), "partial trace is not normalized")
    check(isfinite(norm(derivative)), "composite operator application returned a non-finite result")
    return norm(derivative)
end

function particle()
    position_basis = PositionBasis(-10, 10, 200)
    momentum_basis = MomentumBasis(position_basis)
    position_state = gaussianstate(position_basis, 0.1, 0.3, 1.0)
    to_momentum = transform(momentum_basis, position_basis)
    momentum_state = to_momentum * position_state

    position_operator = position(position_basis)
    momentum_operator = momentum(momentum_basis)
    hamiltonian = LazySum(
        LazyProduct(dagger(to_momentum), momentum_operator^2 / 2, to_momentum),
        position_operator^2,
    )
    derivative = hamiltonian * position_state

    check(isapprox(norm(momentum_state), 1; atol=1e-12), "particle transform changed the state norm")
    check(isapprox(real(expect(momentum_operator, momentum_state)), 0.3; atol=1e-12), "particle momentum is incorrect")
    check(isfinite(norm(derivative)), "lazy particle Hamiltonian returned a non-finite result")
    return norm(derivative)
end

function manybody()
    one_body_basis = NLevelBasis(2)
    many_body_basis = ManyBodyBasis(
        one_body_basis,
        bosonstates(one_body_basis, [0, 1, 2, 3]),
    )
    one_body_hamiltonian = diagonaloperator(one_body_basis, [0.0, 1.0])
    many_body_hamiltonian = manybodyoperator(many_body_basis, one_body_hamiltonian)
    state = basisstate(many_body_basis, [1, 2])
    transferred = transition(many_body_basis, 1, 2) * state
    expected = 2 * basisstate(many_body_basis, [2, 1])

    check(isapprox(expect(many_body_hamiltonian, state), 2; atol=1e-12), "many-body energy is incorrect")
    check(norm(transferred - expected) < 1e-12, "many-body transition returned the wrong state")
    return real(expect(many_body_hamiltonian, state))
end

function superoperator()
    basis = FockBasis(10)
    hamiltonian = number(basis)
    jump = destroy(basis)
    generator = liouvillian(hamiltonian, [jump])
    density = dm(coherentstate(basis, 0.5))
    derivative = generator * density

    check(isapprox(tr(derivative), 0; atol=1e-11), "Lindblad generator is not trace preserving")
    check(isfinite(norm(derivative)), "Lindblad generator returned a non-finite result")
    return norm(derivative)
end

function metrics()
    basis = SpinBasis(1 // 2)
    up = spinup(basis)
    down = spindown(basis)
    state = (up ⊗ down - down ⊗ up) / sqrt(2)
    density = dm(state)
    reduced_spin = ptrace(density, 2)

    check(isapprox(tr(reduced_spin), 1; atol=1e-12), "reduced spin state is not normalized")
    check(isapprox(entropy_vn(reduced_spin), log(2); atol=1e-12), "spin entropy is incorrect")
    check(isapprox(negativity(density, 1), 0.5; atol=1e-12), "spin negativity is incorrect")
    check(isapprox(real(fidelity(reduced_spin, reduced_spin)), 1; atol=1e-12), "self-fidelity is incorrect")
    return real(entropy_vn(reduced_spin))
end

function nlevel()
    basis = NLevelBasis(4)
    state = nlevelstate(basis, 2)
    shift = paulix(basis)
    clock = pauliz(basis)
    transitioned = transition(basis, 3, 2) * state

    check(isapprox(norm(transitioned), 1; atol=1e-12), "N-level transition returned the wrong norm")
    check(isapprox(norm(shift^4 - identityoperator(basis)), 0; atol=1e-12), "N-level shift is not cyclic")
    check(isapprox(norm(clock^4 - identityoperator(basis)), 0; atol=1e-12), "N-level clock is not cyclic")
    return norm(transitioned)
end

function charge()
    basis = ChargeBasis(5)
    state = chargestate(basis, -3)
    shift = expiφ(basis)
    charge_operator = chargeop(basis)
    shifted = shift * state

    check(shifted == chargestate(basis, -2), "charge shift returned the wrong state")
    check(charge_operator * state == -3 * state, "charge operator returned the wrong eigenvalue")
    check(charge_operator * shift - shift * charge_operator == shift, "charge commutator is incorrect")
    return norm(shifted)
end

function time_dependent()
    basis = SpinBasis(1 // 2)
    sx = sigmax(basis)
    sz = sigmaz(basis)
    operator = TimeDependentSum((cos, sin), (sx, sz))
    set_time!(operator, 0.25)
    state = spinup(basis)
    result = operator * state
    expected = (cos(0.25) * sx + sin(0.25) * sz) * state

    check(current_time(operator) == 0.25, "time-dependent operator has the wrong time")
    check(norm(result - expected) < 1e-12, "time-dependent operator returned the wrong state")
    return norm(result)
end

function pauli()
    basis = SpinBasis(1 // 2)
    up = spinup(basis)
    down = spindown(basis)
    gate = dm(up) ⊗ identityoperator(basis) + dm(down) ⊗ sigmax(basis)
    superoperator_gate = SuperOperator(gate)
    chi_gate = ChiMatrix(gate)
    transfer_gate = PauliTransferMatrix(gate)

    check(isapprox(avg_gate_fidelity(superoperator_gate, superoperator_gate), 1; atol=1e-12), "superoperator fidelity is incorrect")
    check(isapprox(avg_gate_fidelity(chi_gate, chi_gate), 1; atol=1e-12), "Chi-matrix fidelity is incorrect")
    check(isapprox(avg_gate_fidelity(transfer_gate, transfer_gate), 1; atol=1e-12), "Pauli-transfer fidelity is incorrect")
    return real(avg_gate_fidelity(transfer_gate, transfer_gate))
end

const SCENARIOS = Dict(
    "fock" => fock,
    "composite" => composite,
    "particle" => particle,
    "manybody" => manybody,
    "superoperator" => superoperator,
    "metrics" => metrics,
    "nlevel" => nlevel,
    "charge" => charge,
    "time_dependent" => time_dependent,
    "pauli" => pauli,
)

length(ARGS) == 1 || error("usage: scenarios.jl SCENARIO")
scenario_name = only(ARGS)
scenario = get(SCENARIOS, scenario_name, nothing)
isnothing(scenario) && error("unknown precompile scenario: $(scenario_name)")

trace_mode = get(ENV, "QOB_PRECOMPILE_TRACE", "")
first_result = if isempty(trace_mode)
    @timed scenario()
elseif trace_mode == "compile" || trace_mode == "dispatch"
    VERSION >= v"1.12" || error("QOB_PRECOMPILE_TRACE requires Julia 1.12 or later")
    Core.eval(
        @__MODULE__,
        Meta.parse("@timed Base.@trace_$(trace_mode) $(scenario_name)()"),
    )
else
    error("QOB_PRECOMPILE_TRACE must be empty, compile, or dispatch")
end
total_seconds = (time_ns() - total_started_ns) / 1.0e9
warm_result = @timed scenario()
compile_time(result) = hasproperty(result, :compile_time) ? result.compile_time : 0.0
recompile_time(result) = hasproperty(result, :recompile_time) ? result.recompile_time : 0.0

println(join((
    "RESULT",
    scenario_name,
    import_seconds,
    first_result.time,
    compile_time(first_result),
    recompile_time(first_result),
    total_seconds,
    warm_result.time,
    compile_time(warm_result),
    recompile_time(warm_result),
), '\t'))

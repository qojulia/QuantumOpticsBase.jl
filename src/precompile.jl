using PrecompileTools

@setup_workload let
    # Fock-space states and operators
    fock_basis = FockBasis(20)
    alpha = 1.2

    @compile_workload begin
        annihilation = destroy(fock_basis)
        state = coherentstate(fock_basis, alpha)
        density = dm(state)
        derivative = -1im * (annihilation + create(fock_basis)) * state

        amplitude = expect(annihilation, state)
        photon_number = expect(number(fock_basis), density)
        @assert isapprox(amplitude, alpha; atol=1e-10)
        @assert isapprox(photon_number, abs2(alpha); atol=1e-9)
        @assert isapprox(tr(density), 1; atol=1e-12)
        @assert isfinite(norm(derivative))
    end
end

@setup_workload let
    # Composite systems, embedding, and partial traces
    fock_basis = FockBasis(20)
    spin_basis = SpinBasis(1 // 2)

    @compile_workload begin
        raising = sigmap(spin_basis)
        lowering = sigmam(spin_basis)
        atom_hamiltonian = 2 * raising * lowering

        annihilation = destroy(fock_basis)
        field_hamiltonian = number(fock_basis)
        interaction = annihilation ⊗ raising + create(fock_basis) ⊗ lowering

        composite_basis = fock_basis ⊗ spin_basis
        hamiltonian =
            embed(composite_basis, 1, field_hamiltonian) +
            embed(composite_basis, 2, atom_hamiltonian) +
            interaction
        initial_state = fockstate(fock_basis, 1) ⊗ spindown(spin_basis)
        derivative = hamiltonian * initial_state
        reduced_field = ptrace(dm(initial_state), 2)

        @assert basis(derivative) == composite_basis
        @assert isapprox(tr(reduced_field), 1; atol=1e-12)
        @assert isfinite(norm(derivative))
    end
end

@setup_workload let
    # Entanglement metrics
    spin_basis = SpinBasis(1 // 2)

    @compile_workload begin
        up = spinup(spin_basis)
        down = spindown(spin_basis)
        state = (up ⊗ down - down ⊗ up) / sqrt(2)
        density = dm(state)
        reduced_spin = ptrace(density, 2)

        entropy = real(entropy_vn(reduced_spin))
        state_negativity = negativity(density, 1)
        self_fidelity = real(fidelity(reduced_spin, reduced_spin))
        @assert isapprox(tr(reduced_spin), 1; atol=1e-12)
        @assert isapprox(entropy, log(2); atol=1e-12)
        @assert isapprox(state_negativity, 0.5; atol=1e-12)
        @assert isapprox(self_fidelity, 1; atol=1e-12)
    end
end

@setup_workload let
    # Lindblad superoperators
    fock_basis = FockBasis(10)

    @compile_workload begin
        hamiltonian = number(fock_basis)
        jump = destroy(fock_basis)
        generator = liouvillian(hamiltonian, [jump])
        density = dm(coherentstate(fock_basis, 0.5))
        derivative = generator * density

        @assert isapprox(tr(derivative), 0; atol=1e-11)
        @assert isfinite(norm(derivative))
    end
end

@setup_workload let
    # Time-dependent operators
    spin_basis = SpinBasis(1 // 2)
    sample_time = 0.25

    @compile_workload begin
        sx = sigmax(spin_basis)
        sz = sigmaz(spin_basis)
        operator = TimeDependentSum((cos, sin), (sx, sz))
        set_time!(operator, sample_time)

        state = spinup(spin_basis)
        result = operator * state
        expected = (cos(sample_time) * sx + sin(sample_time) * sz) * state
        @assert current_time(operator) == sample_time
        @assert norm(result - expected) < 1e-12
    end
end

@setup_workload let
    # Bosonic many-body operators
    one_body_basis = NLevelBasis(2)

    @compile_workload begin
        many_body_basis = ManyBodyBasis(
            one_body_basis,
            bosonstates(one_body_basis, [0, 1, 2, 3]),
        )
        one_body_hamiltonian = diagonaloperator(one_body_basis, [0.0, 1.0])
        many_body_hamiltonian = manybodyoperator(
            many_body_basis,
            one_body_hamiltonian,
        )

        state = basisstate(many_body_basis, [1, 2])
        transferred = transition(many_body_basis, 1, 2) * state
        expected = 2 * basisstate(many_body_basis, [2, 1])
        energy = real(expect(many_body_hamiltonian, state))
        @assert isapprox(energy, 2; atol=1e-12)
        @assert norm(transferred - expected) < 1e-12
    end
end

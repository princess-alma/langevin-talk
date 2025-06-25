# Langevin Dynamics

## Applications in Machine Learning

Alma Ament


---

## From Newton to Langevin


<section style="position: relative; top: -50px;">
  <!-- Portraits positioned in the top-right corner -->
  <div style="position: absolute; top: -50px; right: 0; display: flex; gap: 10px;">
    <img src="newton_portrait.jpg" alt="Isaac Newton" style="width: 100px; height: 120px; object-fit: cover; border-radius: 5px;">
    <img src="langevin_portrait.jpg" alt="Paul Langevin" style="width: 100px; height: 120px; object-fit: cover; border-radius: 5px;">
  </div>

  <!-- Content columns -->
  <div class="columns" style="margin-top: 30px;">
    <div class="column">
      <strong>Newtonian Mechanics</strong>
      <ul>
        <li>Smooth, deterministic motion</li>
        <li>Single trajectory per initial condition</li>
      </ul>
      $$
      \mathbf{F} = m\textbf{a} = m \frac{d^2 \mathbf{x}}{dt^2} = m\ddot{x}
      $$
      $$
      \begin{cases}
      \frac{d\mathbf{x}}{dt} = \mathbf{v}, \\
      \frac{d\mathbf{v}}{dt} = \frac{\mathbf{F}(\mathbf{x}, \mathbf{v}, t)}{m}
      \end{cases}
      $$
    </div>
    <div class="column">
      <strong>Langevin Dynamics</strong>
      <ul>
        <li>Random, jittery motion</li>
        <li>Ensemble of possible paths</li>
      </ul>
      $$
      m \frac{d^2 x}{dt^2} = -\gamma \frac{dx}{dt} - \nabla U(x) + \eta(t)
      $$
      <ul>
      <li>$m$: mass of the particle</li>
      <li>$\gamma$: friction coefficient (damping)</li>
      <li>$U(x)$: potential energy</li>
      <li>$\eta(t)$: stochastic force with $\eta(t) \sim \mathcal{N}\left(0,\, 2D\,\delta(t-t')\right)$</li>
      </ul>
    </div>
  </div>
</section>

---

## Newtonian Mechanics vs Langevin Dynamics

<div style="text-align: center;">
  <video controls width="90%" style="max-width: 100%; height: auto; outline: none; border: none;" onclick="this.blur();">
    <source src="combined_dynamics_1.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

---

## Langevin Dynamics

<div style="text-align: center;">
  <video controls width="90%" style="max-width: 100%; height: auto; outline: none; border: none;" onclick="this.blur();">
    <source src="combined_dynamics_2.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

---

## Overdamped Langevin Dynamics


$$ m \ddot{x} = -\gamma \dot{x} - \nabla V(x) + \eta(t) $$


*  The frictional force ($-\gamma \dot{x}$) is **huge**
*  The inertial force ($m \ddot{x}$) is **negligibly small** in comparison

$$ m \ddot{x} \approx 0 \Rightarrow \gamma \dot{x} = -\nabla V(x) + \eta(t) $$

<div style="text-align: right; margin-top: -100px; margin-right: 20px;">
  <img src="marble_in_honey.png" alt="Marble in honey illustration" style="width: 300px; height: auto; border-radius: 10px;">
</div>

---

## Brownian Motion

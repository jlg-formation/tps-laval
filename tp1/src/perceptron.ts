// ---------------------------------------------------------------------------
// Le perceptron : le plus simple des neurones artificiels.
//
// Il calcule une somme pondérée de ses entrées, ajoute un biais, puis applique
// une fonction d'activation « à seuil » (step) qui renvoie 0 ou 1.
//
//   sortie = 1  si  (w1*x + w2*y + b) >= 0
//   sortie = 0  sinon
//
// La frontière de décision est donc la droite d'équation  w1*x + w2*y + b = 0.
// Apprendre, c'est ajuster w1, w2 et b pour que cette droite sépare les deux
// classes de points.
// ---------------------------------------------------------------------------

/** Un exemple d'entraînement : deux entrées normalisées et la classe attendue. */
export interface TrainingSample {
  x: number; // première entrée (coordonnée normalisée dans [-1, 1])
  y: number; // seconde entrée (coordonnée normalisée dans [-1, 1])
  label: 0 | 1; // classe cible attendue
}

export class Perceptron {
  w1 = 0;
  w2 = 0;
  b = 0;

  /** Nombre total de mises à jour tentées depuis le dernier reset. */
  iterations = 0;

  constructor() {
    this.reset();
  }

  /** Repart de poids aléatoires faibles pour casser la symétrie. */
  reset(): void {
    this.w1 = Math.random() * 0.2 - 0.1;
    this.w2 = Math.random() * 0.2 - 0.1;
    this.b = Math.random() * 0.2 - 0.1;
    this.iterations = 0;
  }

  /** Applique le neurone : renvoie la classe prédite (0 ou 1). */
  predict(x: number, y: number): 0 | 1 {
    const sum = this.w1 * x + this.w2 * y + this.b;
    return sum >= 0 ? 1 : 0;
  }

  /**
   * Règle d'apprentissage du perceptron, sur UN exemple.
   *
   *   erreur = cible - sortie          (vaut -1, 0 ou +1)
   *   wi    += lr * erreur * entrée_i
   *   b     += lr * erreur
   *
   * Si la prédiction est correcte, erreur = 0 : rien ne bouge.
   * Sinon on pousse la droite dans la bonne direction, proportionnellement au
   * taux d'apprentissage `lr`.
   *
   * @returns true si l'exemple était mal classé (donc a provoqué une correction).
   */
  trainStep(sample: TrainingSample, lr: number): boolean {
    const output = this.predict(sample.x, sample.y);
    const error = sample.label - output;

    if (error !== 0) {
      this.w1 += lr * error * sample.x;
      this.w2 += lr * error * sample.y;
      this.b += lr * error;
    }

    this.iterations++;
    return error !== 0;
  }

  /** Compte combien d'exemples sont actuellement mal classés. */
  countErrors(samples: TrainingSample[]): number {
    let errors = 0;
    for (const s of samples) {
      if (this.predict(s.x, s.y) !== s.label) errors++;
    }
    return errors;
  }
}

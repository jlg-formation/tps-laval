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
   * Une ITÉRATION = une passe complète (batch) sur tous les échantillons.
   *
   * On parcourt d'abord TOUS les points pour accumuler le gradient (la somme des
   * corrections souhaitées), puis on applique UNE SEULE mise à jour des poids —
   * c'est la descente de gradient « batch ».
   *
   *   pour chaque point :  erreur = cible - sortie
   *                        g_w1 += erreur * x
   *                        g_w2 += erreur * y
   *                        g_b  += erreur
   *   puis, une seule fois :
   *                        w1 += lr * g_w1 / N
   *                        w2 += lr * g_w2 / N
   *                        b  += lr * g_b  / N
   *
   * On moyenne par N (nombre de points) pour que le taux d'apprentissage garde
   * le même effet quel que soit le nombre d'échantillons.
   *
   * @returns true s'il restait au moins un point mal classé pendant la passe.
   */
  trainEpoch(samples: TrainingSample[], lr: number): boolean {
    if (samples.length === 0) return false;

    let gW1 = 0;
    let gW2 = 0;
    let gB = 0;
    let errors = 0;

    // Phase 1 : on parcourt tous les points et on accumule le gradient.
    for (const s of samples) {
      const output = this.predict(s.x, s.y);
      const error = s.label - output;
      if (error !== 0) errors++;
      gW1 += error * s.x;
      gW2 += error * s.y;
      gB += error;
    }

    // Phase 2 : une seule mise à jour des poids à partir du gradient accumulé.
    const n = samples.length;
    this.w1 += (lr * gW1) / n;
    this.w2 += (lr * gW2) / n;
    this.b += (lr * gB) / n;

    this.iterations++;
    return errors > 0;
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

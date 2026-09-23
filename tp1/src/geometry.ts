// ---------------------------------------------------------------------------
// Conversions entre l'espace « pixels » du canvas et l'espace « normalisé »
// [-1, 1] dans lequel le perceptron travaille.
//
// Travailler en [-1, 1] plutôt qu'en pixels rend l'apprentissage bien plus
// stable : les entrées ont un ordre de grandeur raisonnable, donc un même
// learning rate fonctionne quelle que soit la taille du canvas.
// ---------------------------------------------------------------------------

export interface Vec2 {
  x: number;
  y: number;
}

/** Pixel -> coordonnée normalisée dans [-1, 1]. */
export function toNormalized(px: number, py: number, width: number, height: number): Vec2 {
  return {
    x: (px / width) * 2 - 1,
    y: (py / height) * 2 - 1,
  };
}

/** Coordonnée normalisée [-1, 1] -> pixel. */
export function toPixel(nx: number, ny: number, width: number, height: number): Vec2 {
  return {
    x: ((nx + 1) / 2) * width,
    y: ((ny + 1) / 2) * height,
  };
}

/**
 * Calcule les deux extrémités (en pixels) du segment représentant la droite
 * de décision  w1*x + w2*y + b = 0  à travers tout le canvas.
 *
 * On raisonne dans l'espace normalisé [-1, 1] puis on reconvertit en pixels.
 * Selon l'orientation de la droite, on la paramètre soit par x, soit par y,
 * pour éviter les divisions par des poids proches de zéro.
 *
 * @returns les deux extrémités, ou null si la droite est indéfinie (w1=w2=0).
 */
export function decisionLineEndpoints(
  w1: number,
  w2: number,
  b: number,
  width: number,
  height: number,
): [Vec2, Vec2] | null {
  if (w1 === 0 && w2 === 0) return null;

  let a: Vec2;
  let c: Vec2;

  if (Math.abs(w2) >= Math.abs(w1)) {
    // Droite « plutôt horizontale » : on balaie x de -1 à 1 et on résout y.
    const yAt = (x: number) => -(b + w1 * x) / w2;
    a = { x: -1, y: yAt(-1) };
    c = { x: 1, y: yAt(1) };
  } else {
    // Droite « plutôt verticale » : on balaie y de -1 à 1 et on résout x.
    const xAt = (y: number) => -(b + w2 * y) / w1;
    a = { x: xAt(-1), y: -1 };
    c = { x: xAt(1), y: 1 };
  }

  return [toPixel(a.x, a.y, width, height), toPixel(c.x, c.y, width, height)];
}

export function slugify(value: string): string {
  // Injected bug: consecutive spaces produce repeated dashes.
  return value.trim().toLowerCase().replace(/\s/g, "-");
}


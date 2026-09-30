"""Render a dependency-free SVG validation curve from train.py metrics."""
import argparse
import json
from pathlib import Path


def write_validation_svg(history, output_path):
    """Write an SVG line chart for validation BPB observations."""
    if not history:
        return
    steps = [row['step'] for row in history]
    values = [row['bpb'] for row in history]
    left, top, width, height = 72, 28, 720, 360
    low, high = min(values), max(values)
    padding = max((high-low)*.08, .001)
    low, high = low-padding, high+padding
    max_step = max(steps)
    points = ' '.join(
        f'{left + width * step / max_step:.1f},{top + height * (high-value) / (high-low):.1f}'
        for step, value in zip(steps, values)
    )
    labels = ''.join(
        f'<text x="{left-10}" y="{top + height * fraction + 4:.1f}" text-anchor="end">'
        f'{high - (high-low) * fraction:.4f}</text>'
        for fraction in (0., .5, 1.)
    )
    output_path = Path(output_path)
    output_path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="840" height="440" viewBox="0 0 840 440">
<style>text{{font:14px sans-serif;fill:#263238}} .axis{{stroke:#607d8b}} .grid{{stroke:#cfd8dc;stroke-dasharray:4 4}}</style>
<rect width="840" height="440" fill="#ffffff"/>
<text x="72" y="18">Validation BPB</text>
<line class="axis" x1="72" y1="388" x2="792" y2="388"/><line class="axis" x1="72" y1="28" x2="72" y2="388"/>
<line class="grid" x1="72" y1="28" x2="792" y2="28"/><line class="grid" x1="72" y1="208" x2="792" y2="208"/><line class="grid" x1="72" y1="388" x2="792" y2="388"/>
{labels}<text x="792" y="416" text-anchor="end">step {max_step}</text>
<polyline fill="none" stroke="#00796b" stroke-width="3" points="{points}"/>
{''.join(f'<circle cx="{point.split(',')[0]}" cy="{point.split(',')[1]}" r="4" fill="#d84315"/>' for point in points.split())}
</svg>'''
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('history', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    history = [json.loads(line) for line in args.history.read_text().splitlines() if line]
    output = args.output or args.history.with_name('validation_curve.svg')
    write_validation_svg(history, output)
    print(output)


if __name__ == '__main__':
    main()
The Layout Config Impact Explorer (Layout Config Impact Explorer) provides a detailed breakdown of die dimensions and their corresponding feature impact values across multiple configuration files (currently viewing heat_analysis_v5.json).

Die Dimensions: Includes width and height measurements for 8 dies (G0–G3 and H0–H3).

Feature Impact: Highlights which specific dimension features drive the most impact. Notably, G0_height shows the highest impact value at 0.3078, followed by H2_height (0.1416) and G1_height (0.1105), while other features like H1_width and H3_width show minimal impact near 0.0003.

How to run

python app.py
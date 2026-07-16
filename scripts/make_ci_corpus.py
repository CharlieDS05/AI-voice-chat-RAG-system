"""Generate the synthetic CI test corpus: a small fictional handbook.

Every fact in this document is invented and owned by this repo, so it
can be committed and rebuilt anywhere. Page layout is explicit so the
CI golden set can assert expected_pages deterministically.

Usage: python -m scripts.make_ci_corpus
"""

from pathlib import Path

from fpdf import FPDF

PAGES = [
    # --- page 1 ---
    (
        "Aurora Coffee Company - Employee Handbook\n\n"
        "Chapter 1: Our History\n\n"
        "Aurora Coffee Company was founded in 2011 in Portland, Oregon, by "
        "Elena Vasquez and Tomas Lind. The company began as a single roastery "
        "cart at the Saturday market and opened its first permanent cafe on "
        "Alder Street in 2013. Today Aurora operates 14 cafes across three "
        "cities: Portland, Seattle, and Boise. The company motto is 'Warmth "
        "in every cup.' Aurora sources beans directly from twelve partner "
        "farms in Colombia, Ethiopia, and Guatemala, paying a fixed premium "
        "of 30 percent above the fair-trade baseline price."
    ),
    # --- page 2 ---
    (
        "Chapter 2: Products and Roasting\n\n"
        "Aurora produces three signature roast levels. The Dawn roast is a "
        "light roast with citrus notes, roasted for 9 minutes at 196 degrees "
        "Celsius. The Meridian roast is a medium roast with caramel and "
        "hazelnut notes, roasted for 11 minutes. The Dusk roast is a dark "
        "roast with chocolate notes, roasted for 13 minutes. All beans are "
        "rested for 48 hours after roasting before sale. The best-selling "
        "product is the Meridian roast, which accounts for 45 percent of "
        "all bean sales. Cold brew is steeped for 18 hours."
    ),
    # --- page 3 ---
    (
        "Chapter 3: Policies\n\n"
        "Employees receive a free beverage per shift and a 40 percent "
        "discount on all beans. Customer refunds are honored within 30 days "
        "with a receipt. Baristas complete a 6-week training program called "
        "the Aurora Path before working solo. The company donates 2 percent "
        "of annual profits to watershed restoration projects. Cafes are "
        "closed on two holidays each year: Thanksgiving and New Year's Day."
    ),
]


def main() -> None:
    out_dir = Path("data/ci")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ci_corpus.pdf"

    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    for text in PAGES:
        pdf.add_page()
        pdf.multi_cell(w=0, h=8, text=text)
    pdf.output(str(out_path))
    print(f"✅ CI corpus written to {out_path} ({len(PAGES)} pages)")


if __name__ == "__main__":
    main()

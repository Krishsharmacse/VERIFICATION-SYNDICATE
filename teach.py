import textwrap
from duckduckgo_search import DDGS


class FakeNewsEducator:

    def __init__(self):

        # -------------------------------
        # TRUSTED FACT CHECK DOMAINS
        # -------------------------------
        self.trusted_domains = [
            "site:bbc.com",
            "site:reuters.com",
            "site:snopes.com",
            "site:factcheck.org",
            "site:politifact.com",
            "site:altnews.in",
            "site:boomlive.in",
            "site:thequint.com/news/webqoof",
            "site:indiatoday.in/fact-check",
            "site:factly.in"
        ]

        # -------------------------------
        # HELPLINES
        # -------------------------------
        self.helplines = [
            {
                "name": "National Cyber Crime Reporting Portal",
                "website": "https://cybercrime.gov.in",
                "purpose": "Report cybercrime, deepfakes, and misinformation"
            },
            {
                "name": "Cyber Crime Helpline",
                "number": "1930",
                "purpose": "Emergency cyber fraud assistance"
            },
            {
                "name": "PIB Fact Check",
                "website": "https://factcheck.pib.gov.in",
                "whatsapp": "+91 8799711259",
                "purpose": "Verify government schemes and policies"
            }
        ]

    # -------------------------------
    # DUCKDUCKGO SEARCH
    # -------------------------------
    def duckduckgo_search(self, query, max_results=3):

        results = []

        try:
            with DDGS() as ddgs:

                for r in ddgs.text(query, max_results=max_results):

                    results.append({
                        "title": r.get("title"),
                        "link": r.get("href"),
                        "snippet": r.get("body")
                    })

        except Exception as e:

            results.append({"error": str(e)})

        return results

    # -------------------------------
    # FACT CHECK CLAIM
    # -------------------------------
    def verify_claim(self, claim_text):

        domain_filter = " OR ".join(self.trusted_domains)

        query = f'"{claim_text}" fact check ({domain_filter})'

        return self.duckduckgo_search(query, 5)

    # -------------------------------
    # MISINFORMATION TECHNIQUE DETECTOR
    # -------------------------------
    def detect_misinformation_patterns(self, text):

        patterns = []

        text = text.lower()

        if "shocking" in text or "you won't believe" in text:
            patterns.append("Sensational headline")

        if "share this before it gets deleted" in text:
            patterns.append("Urgency manipulation")

        if "secret truth" in text or "they don't want you to know" in text:
            patterns.append("Conspiracy framing")

        if "forward this to everyone" in text:
            patterns.append("Viral chain message")

        if "miracle cure" in text:
            patterns.append("Medical misinformation")

        return patterns

    # -------------------------------
    # EDUCATION CONTENT
    # -------------------------------
    def get_social_media_awareness(self):

        return [
            "1. Fake news spreads quickly on WhatsApp forwards.",
            "2. Emotional headlines are often used to manipulate readers.",
            "3. Always check the original source of the information.",
            "4. AI can generate fake audio and video (deepfakes).",
            "5. Verify claims using trusted fact-checking websites."
        ]

    # -------------------------------
    # REPORTING GUIDE
    # -------------------------------
    def get_reporting_guide(self):

        return {
            "steps": [
                "1. Take a screenshot of the fake message.",
                "2. Save the URL or media file.",
                "3. Report the content on the social media platform.",
                "4. For serious scams call 1930.",
                "5. Report cybercrime on cybercrime.gov.in."
            ],
            "helplines": self.helplines
        }


# -------------------------------
# SIMPLE CLI
# -------------------------------

def main():

    educator = FakeNewsEducator()

    print("\n🛡 MISINFORMATION EDUCATION TOOL\n")

    while True:

        print("\nMenu")
        print("1 Verify a claim")
        print("2 Detect misinformation patterns")
        print("3 Learn social media safety")
        print("4 Reporting helplines")
        print("5 Exit")

        choice = input("\nEnter choice: ")

        if choice == "1":

            claim = input("\nEnter claim: ")

            results = educator.verify_claim(claim)

            print("\nFact Check Results\n")

            for r in results:

                if "error" in r:
                    print("Search error:", r["error"])
                    continue

                print("\nTitle:", r["title"])
                print(textwrap.fill(r["snippet"], width=80))
                print("Source:", r["link"])

        elif choice == "2":

            text = input("\nEnter suspicious message: ")

            patterns = educator.detect_misinformation_patterns(text)

            if patterns:
                print("\n⚠ Possible misinformation techniques:")
                for p in patterns:
                    print("-", p)
            else:
                print("No obvious patterns detected.")

        elif choice == "3":

            tips = educator.get_social_media_awareness()

            print("\nSocial Media Safety Tips\n")

            for t in tips:
                print(textwrap.fill(t, width=80))
                print()

        elif choice == "4":

            guide = educator.get_reporting_guide()

            print("\nReporting Steps\n")

            for step in guide["steps"]:
                print(step)

            print("\nHelplines\n")

            for h in guide["helplines"]:
                print("\n", h["name"])

                if "website" in h:
                    print("Website:", h["website"])

                if "number" in h:
                    print("Call:", h["number"])

                if "whatsapp" in h:
                    print("WhatsApp:", h["whatsapp"])

                print("Purpose:", h["purpose"])

        elif choice == "5":
            break

        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
"""
Chart Analyzer - Analyze chart images and get trading signals
"""
import anthropic
import base64
import sys
from pathlib import Path


def encode_image(image_path: str) -> tuple[str, str]:
    """Encode image to base64."""
    path = Path(image_path)
    suffix = path.suffix.lower()

    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }

    media_type = media_types.get(suffix, "image/png")

    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    return data, media_type


def analyze_chart(image_path: str, api_key: str = None) -> dict:
    """
    Analyze a chart image and return trading signal.

    Args:
        image_path: Path to chart screenshot
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)

    Returns:
        Dictionary with signal, confidence, and analysis
    """
    import os

    if not api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        return {"error": "No API key. Set ANTHROPIC_API_KEY or pass api_key parameter"}

    client = anthropic.Anthropic(api_key=api_key)

    # Encode image
    image_data, media_type = encode_image(image_path)

    # Analysis prompt
    system_prompt = """You are an expert technical analyst. Analyze the chart image and provide a clear trading recommendation.

Your response MUST follow this exact format:

SIGNAL: [BUY/SELL/HOLD]
CONFIDENCE: [0-100]%
DIRECTION: [LONG/SHORT/NEUTRAL]

KEY OBSERVATIONS:
- [observation 1]
- [observation 2]
- [observation 3]

ENTRY: [price level or "market" or "N/A"]
STOP LOSS: [price level or "N/A"]
TAKE PROFIT: [price level or "N/A"]

REASONING:
[2-3 sentences explaining your analysis]

Be decisive. If there's a clear setup, recommend it. If not, say HOLD."""

    user_prompt = """Analyze this chart and tell me:
1. Should I BUY, SELL, or HOLD?
2. How confident are you (0-100%)?
3. Key support/resistance levels
4. Entry, stop loss, and take profit if trading

Look at: trend, candlestick patterns, any visible indicators, support/resistance."""

    # Call Claude with vision
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ],
        system=system_prompt
    )

    analysis_text = response.content[0].text

    # Parse response
    result = {
        "signal": "HOLD",
        "confidence": 50,
        "analysis": analysis_text
    }

    # Extract signal
    if "SIGNAL: BUY" in analysis_text.upper():
        result["signal"] = "BUY"
    elif "SIGNAL: SELL" in analysis_text.upper():
        result["signal"] = "SELL"

    # Extract confidence
    import re
    conf_match = re.search(r'CONFIDENCE:\s*(\d+)', analysis_text)
    if conf_match:
        result["confidence"] = int(conf_match.group(1))

    return result


def main():
    """CLI interface."""
    if len(sys.argv) < 2:
        print("Usage: python chart_analyzer.py <image_path>")
        print("Example: python chart_analyzer.py chart.png")
        print("\nSet ANTHROPIC_API_KEY environment variable first.")
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    print(f"Analyzing: {image_path}")
    print("-" * 40)

    result = analyze_chart(image_path)

    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    # Print results
    signal = result["signal"]
    confidence = result["confidence"]

    # Color coding for terminal
    if signal == "BUY":
        signal_display = f"\033[92m{signal}\033[0m"  # Green
    elif signal == "SELL":
        signal_display = f"\033[91m{signal}\033[0m"  # Red
    else:
        signal_display = f"\033[93m{signal}\033[0m"  # Yellow

    print(f"\nSIGNAAL: {signal_display}")
    print(f"KINDLUS: {confidence}%")
    print("\n" + "=" * 40)
    print(result["analysis"])


if __name__ == "__main__":
    main()

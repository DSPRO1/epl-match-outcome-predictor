"""
Test script for EPL Predictor API

Usage:
    python test_api.py <API_URL> <API_KEY>

Example:
    python test_api.py https://your-workspace--epl-predictor-fastapi-app.modal.run your-secret-key
"""

import requests
import sys
import json
import os


def test_health(base_url):
    """Test health endpoint."""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)

    response = requests.get(f"{base_url}/health")

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def test_root(base_url):
    """Test root endpoint."""
    print("\n" + "="*60)
    print("Testing Root Endpoint")
    print("="*60)

    response = requests.get(base_url)

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def test_predict_simple(base_url, api_key):
    """Test prediction with minimal input."""
    print("\n" + "="*60)
    print("Testing Prediction (Simple)")
    print("="*60)

    payload = {
        "home_team": "Manchester City",
        "away_team": "Liverpool"
    }

    print(f"Request: {json.dumps(payload, indent=2)}")

    response = requests.post(
        f"{base_url}/predict",
        json=payload,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }
    )

    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")

    return response.status_code == 200


def test_predict_full(base_url, api_key):
    """Test prediction with full feature set."""
    print("\n" + "="*60)
    print("Testing Prediction (Full Features)")
    print("="*60)

    # Manchester City vs Arsenal - realistic features
    payload = {
        "home_team": "Manchester City",
        "away_team": "Arsenal",
        "home_elo": 1850,
        "away_elo": 1800,
        "home_gf_roll": 2.4,
        "home_ga_roll": 0.8,
        "home_pts_roll": 2.6,
        "away_gf_roll": 2.1,
        "away_ga_roll": 0.9,
        "away_pts_roll": 2.4,
        "rest_days_home": 7,
        "rest_days_away": 7
    }

    print(f"Request: {json.dumps(payload, indent=2)}")

    response = requests.post(
        f"{base_url}/predict",
        json=payload,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }
    )

    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")

        print("\n" + "-"*60)
        print("PREDICTION SUMMARY")
        print("-"*60)
        print(f"Match: {result['home_team']} vs {result['away_team']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"\nProbabilities:")
        print(f"  Home/Draw: {result['probabilities']['home_or_draw']:.1%}")
        print(f"  Away Win:  {result['probabilities']['away']:.1%}")
    else:
        print(f"Error: {response.text}")

    return response.status_code == 200


def test_predict_underdog(base_url, api_key):
    """Test prediction with underdog scenario."""
    print("\n" + "="*60)
    print("Testing Prediction (Underdog Scenario)")
    print("="*60)

    # Luton Town vs Manchester City - underdog at home
    payload = {
        "home_team": "Luton Town",
        "away_team": "Manchester City",
        "home_elo": 1420,
        "away_elo": 1850,
        "home_gf_roll": 1.2,
        "home_ga_roll": 2.0,
        "home_pts_roll": 1.0,
        "away_gf_roll": 2.4,
        "away_ga_roll": 0.8,
        "away_pts_roll": 2.6,
        "rest_days_home": 7,
        "rest_days_away": 4
    }

    print(f"Request: {json.dumps(payload, indent=2)}")

    response = requests.post(
        f"{base_url}/predict",
        json=payload,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }
    )

    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")

        print("\n" + "-"*60)
        print("PREDICTION SUMMARY")
        print("-"*60)
        print(f"Match: {result['home_team']} vs {result['away_team']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
    else:
        print(f"Error: {response.text}")

    return response.status_code == 200


def main():
    """Run all tests."""
    if len(sys.argv) < 3:
        print("Usage: python test_api.py <API_URL> <API_KEY>")
        print("\nExample:")
        print("  python test_api.py https://your-workspace--epl-predictor-fastapi-app.modal.run your-secret-key")
        print("\nYou can also set the API key as an environment variable:")
        print("  export EPL_API_KEY=your-secret-key")
        print("  python test_api.py https://your-workspace--epl-predictor-fastapi-app.modal.run $EPL_API_KEY")
        sys.exit(1)

    base_url = sys.argv[1].rstrip('/')
    api_key = sys.argv[2]

    print("="*60)
    print("EPL PREDICTOR API TEST SUITE")
    print("="*60)
    print(f"API URL: {base_url}")
    print(f"API Key: {'*' * (len(api_key) - 4)}{api_key[-4:]}")

    results = {
        "Root Endpoint": test_root(base_url),
        "Health Check": test_health(base_url),
        "Simple Prediction": test_predict_simple(base_url, api_key),
        "Full Prediction": test_predict_full(base_url, api_key),
        "Underdog Prediction": test_predict_underdog(base_url, api_key)
    }

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(results.values())
    total = len(results)

    for test_name, passed_test in results.items():
        status = "✓ PASSED" if passed_test else "✗ FAILED"
        print(f"{test_name:<25} {status}")

    print("-"*60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

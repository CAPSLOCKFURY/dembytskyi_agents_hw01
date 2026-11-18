"""
Завдання 1. Інструменти (Tools) з Pydantic v2 схемами та валідацією.
Доменна задача: Фінансовий та інвестиційний асистент.
"""

from typing import Literal
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Mock-дані для інструментів ──────────────────────────────────────────
EXCHANGE_RATES_TO_USD = {
    "USD": 1.0,
    "EUR": 1.08,
    "UAH": 0.024,  # ~41.67 UAH/USD
    "GBP": 1.27,
    "PLN": 0.25,
}

STOCK_DATABASE = {
    "AAPL": {
        "name": "Apple Inc.",
        "price_usd": 228.50,
        "pe_ratio": 34.2,
        "dividend_yield": 0.44,
        "sector": "Technology",
        "risk_level": "medium",
    },
    "MSFT": {
        "name": "Microsoft Corp.",
        "price_usd": 445.20,
        "pe_ratio": 36.8,
        "dividend_yield": 0.72,
        "sector": "Technology",
        "risk_level": "low",
    },
    "NVDA": {
        "name": "NVIDIA Corp.",
        "price_usd": 138.60,
        "pe_ratio": 52.4,
        "dividend_yield": 0.03,
        "sector": "Semiconductors",
        "risk_level": "high",
    },
    "KO": {
        "name": "The Coca-Cola Company",
        "price_usd": 63.10,
        "pe_ratio": 24.5,
        "dividend_yield": 3.08,
        "sector": "Consumer Defensive",
        "risk_level": "low",
    },
    "VOO": {
        "name": "Vanguard S&P 500 ETF",
        "price_usd": 532.00,
        "pe_ratio": 27.1,
        "dividend_yield": 1.32,
        "sector": "Index ETF",
        "risk_level": "low",
    },
}


# ── 1. Конвертація валют ────────────────────────────────────────────────
class CurrencyConvertInput(BaseModel):
    """Параметри для конвертації валют."""

    model_config = ConfigDict(str_strip_whitespace=True)

    amount: float = Field(
        description="Сума для конвертації (додатне число, наприклад 1000.0)"
    )
    from_currency: str = Field(
        description='Вихідна валюта (трилітерний код ISO: "USD", "EUR", "UAH", "GBP", "PLN")'
    )
    to_currency: str = Field(
        description='Цільова валюта (трилітерний код ISO: "USD", "EUR", "UAH", "GBP", "PLN")'
    )

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Сума для конвертації повинна бути більшою за 0")
        if v > 1_000_000_000:
            raise ValueError("Сума перевищує максимально допустимий ліміт (1 млрд)")
        return round(v, 2)

    @field_validator("from_currency", "to_currency")
    @classmethod
    def validate_currency_code(cls, v: str) -> str:
        code = v.upper().strip()
        if len(code) != 3 or not code.isalpha():
            raise ValueError("Код валюти повинен складатися рівно з 3 латинських літер")
        if code not in EXCHANGE_RATES_TO_USD:
            supported = ", ".join(EXCHANGE_RATES_TO_USD.keys())
            raise ValueError(f"Валюта '{code}' не підтримується. Доступні: {supported}")
        return code


@tool(args_schema=CurrencyConvertInput)
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Конвертувати грошову суму з однієї валюти в іншу за актуальним курсом.

    Використовуйте цей інструмент, коли користувач просить перевести гроші
    між валютами (USD, EUR, UAH, GBP, PLN) або коли потрібно привести витрати/інвестиції
    до єдиної валюти.

    Args:
        amount: Сума для конвертації (> 0).
        from_currency: Код вихідної валюти (наприклад, "UAH").
        to_currency: Код цільової валюти (наприклад, "USD").

    Returns:
        Рядок з результатом конвертації та застосованим обмінним курсом.
    """
    from_code = from_currency.upper()
    to_code = to_currency.upper()

    usd_amount = amount * EXCHANGE_RATES_TO_USD[from_code]
    converted = usd_amount / EXCHANGE_RATES_TO_USD[to_code]
    rate = EXCHANGE_RATES_TO_USD[from_code] / EXCHANGE_RATES_TO_USD[to_code]

    return (
        f"{amount:.2f} {from_code} = {converted:.2f} {to_code} "
        f"(курс: 1 {from_code} = {rate:.4f} {to_code})"
    )


# ── 2. Калькулятор складного відсотка ────────────────────────────────────
class CompoundInterestInput(BaseModel):
    """Параметри для розрахунку складного відсотка (депозиту/інвестиції)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    principal: float = Field(
        description="Початкова сума інвестиції або депозиту (додатне число)"
    )
    annual_rate_percent: float = Field(
        description="Річна відсоткова ставка у відсотках (від 0.1 до 100.0, наприклад 12.5)"
    )
    years: int = Field(
        description="Термін інвестування у роках (ціле число від 1 до 50)"
    )
    monthly_contribution: float = Field(
        default=0.0,
        description="Щомісячне поповнення рахунку (невід'ємне число, за замовчуванням 0)",
    )

    @field_validator("principal")
    @classmethod
    def validate_principal(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Початкова сума інвестиції має бути більшою за 0")
        return v

    @field_validator("annual_rate_percent")
    @classmethod
    def validate_rate(cls, v: float) -> float:
        if not (0.1 <= v <= 100.0):
            raise ValueError("Річна ставка повинна бути в межах від 0.1% до 100%")
        return v

    @field_validator("years")
    @classmethod
    def validate_years(cls, v: int) -> int:
        if not (1 <= v <= 50):
            raise ValueError("Термін інвестування має бути від 1 до 50 років")
        return v

    @field_validator("monthly_contribution")
    @classmethod
    def validate_contribution(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Щомісячне поповнення не може бути від'ємним")
        return v


@tool(args_schema=CompoundInterestInput)
def calculate_compound_interest(
    principal: float,
    annual_rate_percent: float,
    years: int,
    monthly_contribution: float = 0.0,
) -> str:
    """Розрахувати дохідність інвестиції або депозиту за формулою складного відсотка з щомісячною капіталізацією.

    Використовуйте цей інструмент, коли потрібно порахувати майбутню вартість
    вкладу, депозиту чи інвестиційного портфеля через N років із заданою
    річною ставкою та опціональним щомісячним поповненням.

    Args:
        principal: Початкова сума вкладу.
        annual_rate_percent: Річна ставка у % (наприклад, 10.0).
        years: Кількість років (1-50).
        monthly_contribution: Сума щомісячного поповнення.

    Returns:
        Детальний розрахунок фінальної суми, суми внесків та чистого прибутку.
    """
    months = years * 12
    monthly_rate = (annual_rate_percent / 100.0) / 12.0

    balance = principal
    for _ in range(months):
        balance = balance * (1 + monthly_rate) + monthly_contribution

    total_invested = principal + (monthly_contribution * months)
    profit = balance - total_invested

    return (
        f"Підсумок через {years} р. (ставка {annual_rate_percent}% річних): "
        f"Фінальна сума = {balance:.2f} | "
        f"Всього вкладено = {total_invested:.2f} | "
        f"Чистий прибуток = {profit:.2f}"
    )


# ── 3. Ринкова інформація про акції/ETF ─────────────────────────────────
class StockInfoInput(BaseModel):
    """Параметри для отримання ринкових даних про акцію або ETF."""

    model_config = ConfigDict(str_strip_whitespace=True)

    ticker: str = Field(
        description='Біржовий тікер активу (наприклад: "AAPL", "MSFT", "NVDA", "KO", "VOO")'
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        clean = v.upper().strip()
        if not clean or not clean.isalnum() or not (1 <= len(clean) <= 5):
            raise ValueError(
                "Тікер повинен містити від 1 до 5 латинських літер/цифр (наприклад, 'AAPL')"
            )
        return clean


@tool(args_schema=StockInfoInput)
def get_stock_info(ticker: str) -> str:
    """Отримати поточну ціну та фінансові показники акції або ETF за тікером.

    Використовуйте цей інструмент, коли користувач запитує про вартість акцій,
    мультиплікатор P/E, дивідендну дохідність, сектор або рівень ризику компанії/ETF
    (доступні тікери в базі: AAPL, MSFT, NVDA, KO, VOO).

    Args:
        ticker: Біржовий тікер (наприклад, "AAPL", "MSFT", "NVDA", "KO", "VOO").

    Returns:
        Рядок із ціною в USD, P/E, дивідендною дохідністю, сектором та ризиком.
    """
    t = ticker.upper()
    if t not in STOCK_DATABASE:
        available = ", ".join(STOCK_DATABASE.keys())
        return f"Тікер '{t}' не знайдено в базі. Доступні тікери: {available}"

    info = STOCK_DATABASE[t]
    return (
        f"{t} ({info['name']}): ціна = ${info['price_usd']:.2f}, "
        f"P/E = {info['pe_ratio']}, дивіденди = {info['dividend_yield']}%, "
        f"сектор = {info['sector']}, рівень ризику = {info['risk_level']}"
    )


# ── 4. Калькулятор кредиту / іпотеки ────────────────────────────────────
class LoanPaymentInput(BaseModel):
    """Параметри для розрахунку ануїтетного платежу за кредитом."""

    model_config = ConfigDict(str_strip_whitespace=True)

    loan_amount: float = Field(
        description="Сума кредиту (додатне число, наприклад 500000)"
    )
    annual_interest_rate: float = Field(
        description="Річна відсоткова ставка за кредитом (від 0.1 до 100.0)"
    )
    term_months: int = Field(
        description="Строк кредиту в місяцях (від 1 до 360)"
    )

    @field_validator("loan_amount")
    @classmethod
    def validate_loan_amount(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Сума кредиту повинна бути більшою за 0")
        return v

    @field_validator("annual_interest_rate")
    @classmethod
    def validate_interest_rate(cls, v: float) -> float:
        if not (0.1 <= v <= 100.0):
            raise ValueError("Відсоткова ставка має бути від 0.1% до 100%")
        return v

    @field_validator("term_months")
    @classmethod
    def validate_term(cls, v: int) -> int:
        if not (1 <= v <= 360):
            raise ValueError("Строк кредиту має бути від 1 до 360 місяців")
        return v


@tool(args_schema=LoanPaymentInput)
def calculate_loan_payment(
    loan_amount: float, annual_interest_rate: float, term_months: int
) -> str:
    """Розрахувати щомісячний ануїтетний платіж та загальну переплату за кредитом.

    Використовуйте цей інструмент, коли користувач хоче розрахувати умови кредиту,
    іпотеки або розстрочки: щомісячний платіж, загальну суму виплат та переплату.

    Args:
        loan_amount: Сума кредиту.
        annual_interest_rate: Річна ставка у %.
        term_months: Термін у місяцях (1-360).

    Returns:
        Рядок із щомісячним платежем, загальною виплатою та сумою переплати.
    """
    monthly_rate = (annual_interest_rate / 100.0) / 12.0
    factor = (1 + monthly_rate) ** term_months
    monthly_payment = loan_amount * (monthly_rate * factor) / (factor - 1)
    total_payment = monthly_payment * term_months
    overpayment = total_payment - loan_amount

    return (
        f"Кредит {loan_amount:.2f} на {term_months} міс. під {annual_interest_rate}% річних: "
        f"Щомісячний платіж = {monthly_payment:.2f} | "
        f"Загальна сума виплат = {total_payment:.2f} | "
        f"Переплата (відсотки) = {overpayment:.2f}"
    )


# Список усіх інструментів для експорту в агент
ALL_TOOLS = [
    convert_currency,
    calculate_compound_interest,
    get_stock_info,
    calculate_loan_payment,
]


def run_tool_unit_tests() -> None:
    """Unit-тести для перевірки роботи та Pydantic-валідації кожного інструменту."""
    print("=== [Завдання 1] Unit-тестування інструментів та Pydantic-валідації ===")

    # 1. Тести успішного виклику
    res1 = convert_currency.invoke(
        {"amount": 10000, "from_currency": "UAH", "to_currency": "USD"}
    )
    print(f"[OK] convert_currency: {res1}")
    assert "240.00 USD" in res1

    res2 = calculate_compound_interest.invoke(
        {
            "principal": 1000,
            "annual_rate_percent": 10.0,
            "years": 5,
            "monthly_contribution": 100,
        }
    )
    print(f"[OK] calculate_compound_interest: {res2}")
    assert "Фінальна сума" in res2

    res3 = get_stock_info.invoke({"ticker": "AAPL"})
    print(f"[OK] get_stock_info: {res3}")
    assert "Apple Inc." in res3

    res4 = calculate_loan_payment.invoke(
        {"loan_amount": 120000, "annual_interest_rate": 15.0, "term_months": 24}
    )
    print(f"[OK] calculate_loan_payment: {res4}")
    assert "Щомісячний платіж" in res4

    # 2. Тести відхилення некоректних даних (Pydantic validation errors)
    invalid_cases = [
        ("convert_currency (від'ємна сума)", CurrencyConvertInput, {"amount": -50, "from_currency": "USD", "to_currency": "UAH"}),
        ("convert_currency (невідома валюта)", CurrencyConvertInput, {"amount": 100, "from_currency": "BTC", "to_currency": "USD"}),
        ("calculate_compound_interest (ставка > 100%)", CompoundInterestInput, {"principal": 1000, "annual_rate_percent": 150.0, "years": 5}),
        ("get_stock_info (порожній тікер)", StockInfoInput, {"ticker": "   "}),
        ("calculate_loan_payment (строк > 360 міс)", LoanPaymentInput, {"loan_amount": 50000, "annual_interest_rate": 12.0, "term_months": 500}),
    ]

    for label, model_cls, payload in invalid_cases:
        try:
            model_cls(**payload)
            raise AssertionError(f"Валідація мала відхилити: {label}")
        except Exception as e:
            print(f"[Очікувана помилка валідації] {label}: {e.errors()[0]['msg'] if hasattr(e, 'errors') else e}")

    print("=== Усі unit-тести інструментів успішно пройдено! ===\n")


if __name__ == "__main__":
    run_tool_unit_tests()

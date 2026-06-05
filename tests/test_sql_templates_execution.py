import pandas as pd

from src.database.queries import read_sql_query
from src.llm.semantic_query_router import QUERY_TEMPLATES
from src.llm.sql_guard import validate_sql_query


def is_valid_sql(sql: str) -> bool:
    """
    Helper to support SQL guard return formats:
    - tuple: (bool, message)
    - dict: {"is_valid": bool, "message": str}
    - string: "SQL query is valid."
    """

    validation_result = validate_sql_query(sql)

    if isinstance(validation_result, tuple):
        return validation_result[0]

    if isinstance(validation_result, dict):
        return validation_result.get("is_valid", False)

    return str(validation_result).strip() == "SQL query is valid."


def test_all_sql_templates_pass_sql_guard():
    for template in QUERY_TEMPLATES:
        sql = template["sql"]

        assert is_valid_sql(sql) is True, (
            f"SQL template failed guard validation: {template['intent']}"
        )


def test_all_sql_templates_execute_successfully():
    for template in QUERY_TEMPLATES:
        sql = template["sql"]

        result_df = read_sql_query(sql)

        assert isinstance(result_df, pd.DataFrame), (
            f"SQL template did not return a DataFrame: {template['intent']}"
        )


def test_all_sql_templates_have_required_fields():
    for template in QUERY_TEMPLATES:
        assert "intent" in template
        assert "keywords" in template
        assert "sql" in template

        assert isinstance(template["intent"], str)
        assert isinstance(template["keywords"], list)
        assert isinstance(template["sql"], str)

        assert template["intent"].strip()
        assert len(template["keywords"]) > 0
        assert template["sql"].strip().lower().startswith("select")


def test_all_sql_templates_return_limited_results():
    for template in QUERY_TEMPLATES:
        sql = template["sql"].lower()

        assert "limit" in sql, (
            f"SQL template should include LIMIT: {template['intent']}"
        )
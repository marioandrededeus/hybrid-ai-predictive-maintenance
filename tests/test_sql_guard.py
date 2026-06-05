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


def test_allows_simple_select_query():
    sql = "SELECT * FROM scenarios LIMIT 10;"

    assert is_valid_sql(sql) is True


def test_allows_select_with_join():
    sql = """
        SELECT
            s.scenario_label,
            AVG(sf.anomaly_score) AS avg_anomaly_score
        FROM spectral_features sf
        JOIN vibration_measurements vm
            ON sf.measurement_id = vm.measurement_id
        JOIN scenarios s
            ON vm.scenario_id = s.scenario_id
        GROUP BY s.scenario_label
        LIMIT 20;
    """

    assert is_valid_sql(sql) is True


def test_blocks_drop_statement():
    sql = "DROP TABLE scenarios;"

    assert is_valid_sql(sql) is False


def test_blocks_delete_statement():
    sql = "DELETE FROM scenarios WHERE scenario_id = 1;"

    assert is_valid_sql(sql) is False


def test_blocks_update_statement():
    sql = "UPDATE scenarios SET severity_level = 'high' WHERE scenario_id = 1;"

    assert is_valid_sql(sql) is False


def test_blocks_insert_statement():
    sql = "INSERT INTO scenarios (scenario_name) VALUES ('test');"

    assert is_valid_sql(sql) is False


def test_blocks_create_statement():
    sql = "CREATE TABLE test_table (id INTEGER);"

    assert is_valid_sql(sql) is False


def test_blocks_alter_statement():
    sql = "ALTER TABLE scenarios ADD COLUMN test_column TEXT;"

    assert is_valid_sql(sql) is False


def test_blocks_multiple_statements_with_dangerous_command():
    sql = "SELECT * FROM scenarios; DROP TABLE scenarios;"

    assert is_valid_sql(sql) is False

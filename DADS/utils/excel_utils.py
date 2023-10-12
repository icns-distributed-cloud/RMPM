import xlrd
import xlwt
from xlutils.copy import copy

def create_excel_xsl(path, sheet_name, value):
    """
    값에 기반하여 엑셀 테이블과 시트를 만듭니다.
    :param 경로: 테이블 경로
    :param 시트_이름: 시트 이름
    :param 값: 열 제목, 열 제목 형식은 다음과 같습니다.
    :return: None

    값 = [["feature1", "feature2", "feature3"....]]
    """
    index = len(value)
    try:
        with xlrd.open_workbook(path) as workbook:
            workbook = copy(workbook)
            # worksheet = workbook.sheet_by_name(sheet_name)
            worksheet = workbook.add_sheet(sheet_name)  # 통합 통계 워크북에서 새로운 테이블 생성
            for i in range(len(value[0])):
                worksheet.col(i).width = 256 * 30  # Set the column width
            for i in range(0, index):
                for j in range(0, len(value[i])):
                    worksheet.write(i, j, value[i][j])
            workbook.save(path)
            print("xls 형식의 테이블이 성공적으로 생성되었습니다.")
    except FileNotFoundError:
        workbook = xlwt.Workbook()  # 新建一个工作簿
        worksheet = workbook.add_sheet(sheet_name)  # 통합 통계 워크북에서 새로운 테이블 생성
        for i in range(len(value[0])):
            worksheet.col(i).width = 256 * 30  # Set the column width
        for i in range(0, index):
            for j in range(0, len(value[i])):
                worksheet.write(i, j, value[i][j])
        workbook.save(path)
        print("xls 형식의 테이블이 성공적으로 생성되었습니다.")


def write_excel_xls_append(path, sheet_name, value):
    """
    将value值写入到指定的excel表格中
    :param path: 表格路径
    :param sheet_name: sheet名称
    :param value: 新增一列，形式如下
    :return: None

    value = [["feature1", "feature2", "feature3"....]]
    """
    index = len(value)
    workbook = xlrd.open_workbook(path)
    worksheet = workbook.sheet_by_name(sheet_name)

    rows_old = worksheet.nrows  # 기존에 존재하는 표의 데이터 행 수를 가져옵니다.
    new_workbook = copy(workbook)  # xlrd 객체를 복사하여 xlwt 객체로 변환합니다
    new_worksheet = new_workbook.get_sheet(sheet_name)

    for i in range(len(value[0])):
        new_worksheet.col(i).width = 256 * 30  # Set the column width

    for i in range(0, index):
        for j in range(0, len(value[i])):
            new_worksheet.write(i + rows_old, j, value[i][j])

    new_workbook.save(path)  # 워크북 저장
    print("xls 형식의 테이블에 데이터를 추가하여 성공적으로 쓰였습니다!")


def sheet_exists(path, sheet_name):
    """
    주어진 엑셀 파일에서 주어진 시트 이름이 존재하는지 여부를 확인합니다.
    :param path: 엑셀 파일 경로
    :param sheet_name: sheet이름
    :return: True or False 시트 존재 여부
    """
    try:
        workbook = xlrd.open_workbook(path)
        worksheet = workbook.sheet_by_name(sheet_name)
        if worksheet is None:
            return False
    except Exception:
        return False


def read_excel_xls(path, sheet_name):
    """
    엑셀 테이블에서 데이터를 표시합니다.
    :param path: 테이블 경로
    :param sheet_name: sheet이름
    :return:
    """
    workbook = xlrd.open_workbook(path)  # 통합 문서 열기
    worksheet = workbook.sheet_by_name(sheet_name)  # 통합 문서에서 모든 시트 가져오기
    for i in range(0, worksheet.nrows):
        for j in range(0, worksheet.ncols):
            print(worksheet.cell_value(i, j), "\t", end="")  # 행별 열별 데이터 읽기
        print()


def get_excel_data(path, sheet_name, col_name):
    """
    엑셀 시트에서 지정된 열의 데이터를 읽어옵니다.
    :param path: 파일 경로
    :param sheet_name: 시트 이름
    :param col_name: 시트 내의 열 이름 또는 속성 이름
    :return: 해당 데이터의 목록
    """
    workbook = xlrd.open_workbook(path)  # 통합 문서 열기
    worksheet = workbook.sheet_by_name(sheet_name)  #  통합 문서에서 모든 시트 가져오기

    col_index = -1
    for j in range(0, worksheet.ncols):
        if worksheet.cell_value(0, j) == col_name:
            col_index = j
    if col_index == -1:
        print("no matched col name")
        return None

    data = []
    for i in range(1, worksheet.nrows):
        for j in range(0, worksheet.ncols):
            if j == col_index:
                data.append(worksheet.cell_value(i, j))
    return data
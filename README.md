# MCP를 이용한 보고서 작성

여기에서는 MCP로 agent를 생성하여 다양한 보고서를 생성하는 방법에 대해 설명합니다.

## Workflow를 이용한 비용 분석

비용분석을 위한 Workflow는 아래와 같이 정의합니다. [AWS Cost Explorer]를 이용해, 서비스별, 리전별, 기간별 사용량 데이터를 가져옵니다. 이후 정해진 서식에 맞추어 보고서를 생성합니다. 생성된 보고서에 부족한 부분은 MCP를 이용해 얻어진 aws document, aws cli를 이용해 결과를 업데이트합니다.

![image](https://github.com/user-attachments/assets/6851465d-1365-4b50-b873-eab2538bf552)

비용 분석에 대한 리포트는 아래와 같이 주어진 [cost_insight.md](./application/aws_cost/cost_insight.md)는 아래와 같습니다.

```text
당신의 AWS solutions architect입니다.
다음의 Cost Data을 이용하여 user의 질문에 답변합니다.
모르는 질문을 받으면 솔직히 모른다고 말합니다.
답변의 이유를 풀어서 명확하게 설명합니다.

다음 항목들에 대해 분석해주세요:
1. 주요 비용 발생 요인
2. 비정상적인 패턴이나 급격한 비용 증가
3. 비용 최적화가 가능한 영역
4. 전반적인 비용 추세와 향후 예측

분석 결과를 다음과 같은 형식으로 제공해주세요:
## 주요 비용 발생 요인
- [구체적인 분석 내용]

## 이상 패턴 분석
- [비정상적인 비용 패턴 설명]

## 최적화 기회
- [구체적인 최적화 방안]

## 비용 추세
- [추세 분석 및 예측]
```

이때의 AWS 서비스 사용량은 아래와 같습니다.

![image](https://github.com/user-attachments/assets/94dfc329-67f4-409e-8dc6-a912e7a512ed)

리전별 사용량은 아래와 같이 얻을 수 있습니다.

![image](https://github.com/user-attachments/assets/029b6874-ed52-4185-a3b3-e1288009b812)

일자별 사용량은 아래와 같습니다.

![image](https://github.com/user-attachments/assets/4a565709-2f0e-4265-864a-06e0f933c5c5)

이를 분석한 최종 결과를 아래와 같습니다.

![image](https://github.com/user-attachments/assets/27e4dee4-325d-48b1-b4fb-d4aa4a388a6b)

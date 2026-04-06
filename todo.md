## open ai key 사용을 어떻게, KMS 등 AWS 의 리소스 사용

## docker 로 구성 하는데, 회사내부 용도 뿐 아니라 SaaS 구성 외부 사용자에게 플랫폼으로 제공하고자 하는데, AI 리소스가 용량이 큰데 어떻게 SaaS 로 구성할지
```
아래대로 만들면 됩니다.

1. IAM Role 이름 예시
github-actions-ecr-push-role

ARN 예시:

arn:aws:iam::123456789012:role/github-actions-ecr-push-role
이 값을 GitHub Secret AWS_ROLE_ARN에 넣습니다.

2. Trust Policy
GitHub Actions OIDC용 신뢰 정책입니다. YOUR_ORG, YOUR_REPO를 바꿔서 Role에 설정하세요.

{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": [
            "repo:YOUR_ORG/YOUR_REPO:ref:refs/heads/main",
            "repo:YOUR_ORG/YOUR_REPO:pull_request"
          ]
        }
      }
    }
  ]
}
보통 푸시만 허용하려면 pull_request는 빼도 됩니다.

3. ECR Push Policy
이 정책을 Role에 붙이세요.

{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:BatchGetImage",
        "ecr:CompleteLayerUpload",
        "ecr:DescribeRepositories",
        "ecr:InitiateLayerUpload",
        "ecr:PutImage",
        "ecr:UploadLayerPart"
      ],
      "Resource": [
        "arn:aws:ecr:ap-northeast-2:123456789012:repository/ai-meeting-summarizer-api",
        "arn:aws:ecr:ap-northeast-2:123456789012:repository/ai-meeting-summarizer-web"
      ]
    }
  ]
}
리전, 계정 ID, 리포지토리 이름은 실제 값으로 바꾸세요.

4. GitHub에 넣을 값
Settings > Secrets and variables > Actions

Secret:

AWS_ROLE_ARN
Variables:

AWS_REGION 예: ap-northeast-2
ECR_REGISTRY 예: 123456789012.dkr.ecr.ap-northeast-2.amazonaws.com
ECR_API_REPOSITORY 예: ai-meeting-summarizer-api
ECR_WEB_REPOSITORY 예: ai-meeting-summarizer-web
5. 참고
OIDC Provider가 AWS 계정에 먼저 있어야 합니다.
Provider URL:

https://token.actions.githubusercontent.com
Audience:

sts.amazonaws.com
원하시면 다음 답변에서 AWS Console 기준으로 “어디 클릭해서 만드는지” 순서대로 적어드리거나, AWS CLI 명령으로 바로 만들 수 있게 스크립트 형태로 드리겠습니다.
```


## API Hub 를 이용해서 회의록 자동화 모듈 가능한지 - https://aihub.or.kr/
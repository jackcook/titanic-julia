using DataFrames
using DecisionTree
using ScikitLearn

fillna!(dv::DataVector, value::Any) = dv[isna.(dv)] = value

function clean!(dataset)
    dataset[:Title] = 0

    for i in 1:size(dataset, 1)
        title = match(r" ([A-Za-z]+)\.", dataset[:Name][i]).captures[1]

        for t in ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"]
            title = replace(title, t, "Rare")
        end

        title = replace(title, "Mlle", "Miss")
        title = replace(title, "Ms", "Miss")
        title = replace(title, "Mme", "Mrs")

        dataset[:Title][i] = get(Dict(
            "Mr" => 1,
            "Miss" => 2,
            "Mrs" => 3,
            "Master" => 4,
            "Rare" => 5
        ), title, 0)
    end

    dataset[:Sex] = convert(DataArray{Int64}, map(x -> get(Dict("female" => 1, "male" => 0), x, 0), dataset[:Sex]))

    for i in 0:1
        for j in 0:2
            guess_df = dropna(dataset[(dataset[:Sex] .== i) .& (dataset[:Pclass] .== j + 1), :][:Age])
            age_guess = convert(Int64, round((median(guess_df) / 0.5 + 0.5) * 0.5))
            dataset[:Age][isna.(dataset[:Age]) .& (dataset[:Sex] .== i) .& (dataset[:Pclass] .== j + 1)] = age_guess
        end
    end

    dataset[:Age] = convert(DataArray{Int64}, round(dataset[:Age]))

    dataset[:Age][(dataset[:Age] .<= 16)] = 0
    dataset[:Age][(dataset[:Age] .> 16) .& (dataset[:Age] .<= 32)] = 1
    dataset[:Age][(dataset[:Age] .> 32) .& (dataset[:Age] .<= 48)] = 2
    dataset[:Age][(dataset[:Age] .> 48) .& (dataset[:Age] .<= 64)] = 3
    dataset[:Age][(dataset[:Age] .> 64)] = 4

    dataset[:IsAlone] = 0
    dataset[:IsAlone][(dataset[:SibSp] .== 0) .& (dataset[:Parch] .== 0)] = 1

    freq_port = mode(dropna(dataset[:Embarked]))
    fillna!(dataset[:Embarked], freq_port)

    dataset[:Port] = 0

    for i in 1:size(dataset, 1)
        dataset[:Port][i] = get(Dict(
            "S" => 0,
            "C" => 1,
            "Q" => 2
        ), dataset[:Embarked][i], 0)
    end

    delete!(dataset, :Embarked)

    median_fare = median(dropna(dataset[:Fare]))
    dataset[:Fare][isna.(dataset[:Fare])] = median_fare

    delete!(dataset, :Cabin)
    delete!(dataset, :Name)
    delete!(dataset, :Parch)
    delete!(dataset, :SibSp)
    delete!(dataset, :Ticket)

    return dataset
end

train_df = clean!(readtable("./input/train.csv"))
delete!(train_df, :PassengerId)

test_df = clean!(readtable("./input/test.csv"))
passenger_ids = test_df[:PassengerId]

Y_train = train_df[:Survived]
X_train = delete!(train_df, :Survived)

X_test = delete!(test_df, :PassengerId)

model = RandomForestClassifier()
DecisionTree.fit!(model, Array(X_train), Array(Y_train))

Y_pred = ScikitLearn.predict(model, Array(X_test))

submission = DataFrame(PassengerId = passenger_ids, Survived = Y_pred)
writetable("./output/submission.csv", submission)

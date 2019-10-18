const brain = require('brain.js');
const csv = require('csvtojson');
const express = require('express');

const net = new brain.NeuralNetwork()

startup();

async function startup() {
    await createNeuralNetworkModel();
    startAPI();
}

function startAPI() {
    console.log('Starting API...');
    var api = express();

    api.get('/predict', async (request, response) => {
        if (!request.query || !request.query.hour || !request.query.t)
            response.status(400).send('Either hour or temperature missing in query parameters.');

        console.log(`Processing prediction with parameters - hour = ${request.query.hour}, t = ${request.query.t}`);

        const output = net.run( { hour: request.query.hour, t: request.query.t } );
        response.send({
            prediction: getHighest(output),
            fullOutput: output
        });
    });

    api.listen({ port: process.env.PORT || 3000 }, () => {
        console.log(`API running.`);
    });
}

async function createNeuralNetworkModel() {
    const data = await LoadCSV();

    /* 
        CSV file contains three columns. First column is hour in day, second is temperature at that hour, 
        and third is co2 level at that hour and temperature.

        Therefore inputs are hour and temperature, and outputs are co2 levels.
    */

    const trainingData = generateTrainingData(data);

    console.log('Training neural network...');
    net.train(trainingData, { iterations: 1000000, learningRate: 0.1 })

    console.log('Neural network model ready to use.');
}

function getHighest(jsonResults) {
    let high = 0;
    let highName = '';

    for(let i = 0; i < Object.keys(jsonResults).length; i++) {
        if (Object.values(jsonResults)[i] > high) {
            high = Object.values(jsonResults)[i];
            highName = Object.keys(jsonResults)[i];
        }
    }

    return { value: highName, probability: high};
}

async function LoadCSV() {
   console.log('Loading training data...');
   return await csv({noheader: true, headers: ['hour','t','co2']}).fromFile('./data.csv');
}


function generateTrainingData(jsonArray) {
    let result = [];
    jsonArray.forEach(el => {
        let inputs = {}, outputs = {};
        const nrOfKeys = Object.keys(el).length;
        for(let i = 0; i < nrOfKeys; i++) {
            if(i < nrOfKeys-1) inputs[ Object.keys(el)[i] ] = parseInt ( Object.values(el)[i] );
            else outputs[ Object.values(el)[i] ] = 1;
        }
        result.push( { input: inputs, output: outputs } );
    });
    return result;
}

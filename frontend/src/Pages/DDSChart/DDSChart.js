import React, { Component } from 'react';
import APIServices from '../../API/apiservices';
import eventApi from '../../API/eventApi'
import TopMenuBar from '../../Component/TopMenuBar';
import Loader from '../../Component/Loader';
import DrawDDSChart from '../../Component/DDSChart';
import $ from 'jquery';
import Switch from '@material-ui/core/Switch';
import FormGroup from '@material-ui/core/FormGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Access from "../../Constants/accessValidation";
import api from '../../API/api'
import './DDSChart.scss';
import Slider from '@material-ui/core/Slider';
import Spinners from "../../spinneranimation";
import _ from 'lodash';
import Swal from 'sweetalert2';
import cookieStorage from '../../Constants/cookie-storage';


const apiServices = new APIServices();

// const urlSearch = new URLSearchParams(window.location.search)


// let startDate = urlSearch.get('startDate')
// let endDate = urlSearch.get('endDate')
// let regionId = urlSearch.get('regionId')
// let countryId = urlSearch.get('countryId')
// let cityId = urlSearch.get('cityId')

// regionId = regionId ? regionId.includes(',') ? `'${regionId.split(',').join("','")}'` : regionId === 'Null' ? regionId : `'${regionId}'` : ''
// countryId = countryId ? countryId.includes(',') ? `'${countryId.split(',').join("','")}'` : countryId === 'Null' ? countryId : `'${countryId}'` : ''
// cityId = cityId ? cityId.includes(',') ? `'${cityId.split(',').join("','")}'` : cityId === 'Null' ? cityId : `'${cityId}'` : ''

class DDSChart extends Component {
    constructor(props) {
        super(props);
        this.state = {
            revenue: false,
            yaxisname: 'Passenger',
            categories: [],
            dataset: [],
            regionId: null,
            cityId: null,
            countryId: null,
            startDate: null,
            endDate: null,
            loading: false
        };
    }

    sendEvent = (token) => {
        var eventData = {
            event_id: "1",
            description: "User viewed DDS Chart Page",
            where_path: "/DDSChart",
            page_name: "DDS Chart Page"
        }
        eventApi.sendEventWithHeader(eventData, token)
    }

    componentDidMount() {
        const token = cookieStorage.getCookie('Authorization')
        this.sendEvent(token);
    }

    getPOSSelectedGeographicalDetails = ($event) => {
        console.log($event, 'mymy')
        this.setState({
            regionId: $event.regionSelected,
            countryId: $event.countrySelected,
            cityId: $event.citySelected,
            startDate: $event.startDate,
            endDate: $event.endDate,
        }, () => this.getDDSChartData())
    }

    getDDSChartData = () => {
        var self = this;
        const { startDate, endDate, regionId, countryId, cityId, yaxisname } = this.state;

        self.showLoader();
        apiServices.getDDSChartData(startDate, endDate, regionId, countryId, cityId, yaxisname).then(function (result) {
            self.hideLoader();
            if (result) {
                var categories = result.categories;
                var data = result.dataset.filter((d, i) => {
                    d['priority'] = i + 10000
                    if (d.seriesname === 'MH') {
                        d.priority = 1
                    }
                    if (d.seriesname === 'Industry') {
                        d.priority = 2
                    }
                    return d;
                })
                var dataset = data.sort((a, b) => a.priority - b.priority)
                self.setState({ dataset, categories })
            }
        });
    }

    toggleChecked(e) {
        this.setState({ revenue: e.target.checked })
        if (e.target.checked) {
            this.setState({ yaxisname: 'Revenue' }, () => this.getDDSChartData())
        } else {
            this.setState({ yaxisname: 'Passenger' }, () => this.getDDSChartData())
        }
    }

    showLoader = () => {
        this.setState({ loading: true })
        $("#loaderImage").addClass("loader-visible")
    }

    hideLoader = () => {
        this.setState({ loading: false })
        $("#loaderImage").removeClass("loader-visible")
        $(".x_panel").addClass("opacity-fade");
        $(".top-buttons").addClass("opacity-fade");
    }


    render() {
        const { revenue, yaxisname, categories, dataset, loading } = this.state;
        return (
            <div>
                <Loader />
                <TopMenuBar dashboard={true} getPOSSelectedGeographicalDetails={this.getPOSSelectedGeographicalDetails} {...this.props} />
                <div className='ddschart'>
                    <div className='add'>
                        <h2>DDS Chart</h2>
                        <FormGroup>
                            Passenger
                            <FormControlLabel
                                control={<Switch size="small" checked={revenue} onChange={(e) => this.toggleChecked(e)} />}
                            />
                            Revenue
                        </FormGroup>
                    </div>
                    <div className='ddschart-form'>
                        {!loading ? dataset.length === 0 ? <h4 style={{ textAlign: 'center' }}>No Data to Show</h4> : <DrawDDSChart yaxisname={yaxisname} categories={categories} dataset={dataset} /> : ''}
                        {dataset.length > 0 ? <div className='hide-watermark'></div> : ''}
                    </div>
                </div>
            </div>
        );
    }
}

export default DDSChart;
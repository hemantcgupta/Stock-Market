import React, { Component } from 'react';
import APIServices from '../../API/apiservices';
import Chart from "react-google-charts";
import eventApi from '../../API/eventApi'
import TopMenuBar from '../../Component/TopMenuBar';
import Loader from '../../Component/Loader';
import $ from 'jquery';
import Access from "../../Constants/accessValidation";
import api from '../../API/api'
import './TargetRational.scss';
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

class TargetRational extends Component {
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
            description: "User viewed Target Rational Page",
            where_path: "/targetRational",
            page_name: "Target Rational Page"
        }
        // eventApi.sendEventWithHeader(eventData, token)
    }

    componentDidMount() {
        const token = cookieStorage.getCookie('Authorization')
        this.sendEvent(token);
    }

    getPOSSelectedGeographicalDetails = ($event) => {
        this.setState({
            regionId: $event.regionSelected,
            countryId: $event.countrySelected,
            cityId: $event.citySelected,
            startDate: $event.startDate,
            endDate: $event.endDate,
        }, () => this.getTargetRationalData())
    }

    getTargetRationalData = () => {
        // var self = this;
        // const { startDate, endDate, regionId, countryId, cityId, yaxisname } = this.state;

        // self.showLoader();
        // apiServices.getTargetRationalData(startDate, endDate, regionId, countryId, cityId, yaxisname).then(function (result) {
        //     self.hideLoader();
        //     if (result) {
        //         var categories = result.categories;
        //         var data = result.dataset.filter((d, i) => {
        //             d['priority'] = i + 50
        //             if (d.seriesname === 'MH') {
        //                 d.priority = 1
        //             }
        //             if (d.seriesname === 'Industry') {
        //                 d.priority = 2
        //             }
        //             return d;
        //         })
        //         var dataset = data.sort((a, b) => a.priority - b.priority)
        //         self.setState({ dataset, categories })
        //     }
        // });
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

    renderSection1() {
        return (
            <div className='section1'>
                <div className='network'>
                    <div className='rounded-box'>
                        <div className='top-rb'>
                            <p>Network</p>
                        </div>
                        <div className='content'>
                            <span>Revenue</span>
                            <i className='fa fa-arrow-up'></i>
                            <span>1.1M (594)</span>
                        </div>
                    </div>
                    <div class="triangle-right"></div>
                </div>
            </div>
        )
    }

    renderSection2() {
        return (
            <div className='section2'>
                <div className='rev-impact-combination'>
                    <div className='rounded-box-main'>
                        <div className='top-rb'>
                            <p>Revenue Impact due to</p>
                        </div>
                        <div className='inner-rounded-box'>
                            <div className='top-rb content' style={{ flexDirection: 'row', paddingBottom: '0px', fontSize: '14px' }}>
                                <span className='key'>Passenger (2000)</span>
                                <span className='value'><i className='fa fa-arrow-up'></i>0.6M (2%)</span>
                            </div>
                            <div className='content'>
                                <div className='sub-content'>
                                    <span className='key'>Traffic Growth</span>
                                    <span className='value'><i className='fa fa-arrow-up'></i>20%</span>
                                </div>
                                <div className='sub-content'>
                                    <span className='key'>Capacity Growth</span>
                                    <span className='value'><i className='fa fa-arrow-up'></i>30%</span>
                                </div>
                                <div className='sub-content'>
                                    <span className='key'>Market Growth</span>
                                    <span className='value'><i className='fa fa-arrow-up'></i>20%</span>
                                </div>
                                <div className='sub-content'>
                                    <span className='key'>Market Share Growth</span>
                                    <span className='value'><i className='fa fa-arrow-up'></i>20%</span>
                                </div>
                            </div>
                        </div>
                        <div className='inner-rounded-box'>
                            <div className='content'>
                                <div className='sub-content'>
                                    <span className='key'>Exchange Rate</span>
                                    <span className='value'><i className='fa fa-arrow-up'></i>2M (20%)</span>
                                </div>
                            </div>
                        </div>
                        <div className='inner-rounded-box'>
                            <div className='top-rb content' style={{ flexDirection: 'row', paddingBottom: '0px', fontSize: '14px' }}>
                                <span className='key'>Average Fare (2000)</span>
                                <span className='value'><i className='fa fa-arrow-up'></i>0.3M (1%)</span>
                            </div>
                            <div className='content'>
                                <div className='sub-content'>
                                    <span className='key'>Average Fare Growth</span>
                                    <span className='value'><i className='fa fa-arrow-up'></i>10%</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        )
    }

    renderSection3() {
        return (
            <div className='section3'>
                <div className='main-box'>
                    <div class="triangle-right"></div>
                    <div className='rounded-box-main'>
                        <div className='top-rb'>
                            <p>Revenue Impact due to Passengers</p>
                        </div>
                        <div style={{ display: 'flex' }}>
                            <div className='inner-rounded-box'>
                                <div className='content'>
                                    <div className='sub-content'>
                                        <span className='key'>Existing OD</span>
                                        <span className='value'><i className='fa fa-arrow-up'></i>0.4M (1200)</span>
                                    </div>
                                    <div className='sub-content'>
                                        <span className='key'>New OD</span>
                                        <span className='value'><i className='fa fa-arrow-up'></i>0.2M (800)</span>
                                    </div>
                                </div>
                            </div>
                            <div className='piechart-main'>
                                <div className='piechart'>
                                    <Chart
                                        chartType="PieChart"
                                        // loader={<div>Loading Chart</div>}
                                        data={[
                                            ['Task', 'Hours'],
                                            ['New OD', 200000],
                                            ['Existing OD', 400000]
                                        ]}
                                        options={{
                                            // Just add this option
                                            // title: 'My Daily Activities',
                                            'width': 300,
                                            'height': 200,
                                            is3D: true,
                                            legend: 'none',
                                            backgroundColor: 'transparent',
                                            colors: ['#9a1b99', '#ed9720', '#dc392a', '#46961a', '#3566cc']
                                        }}
                                        rootProps={{ 'data-testid': '2' }}
                                    />
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div className='main-box' style={{ marginBottom: '15px' }}>
                    <div class="triangle-right"></div>
                    <div className='rounded-box-main'>
                        <div className='top-rb'>
                            <p>Revenue Impact due to Average Fare</p>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center' }}>
                            <div className='inner-rounded-box' style={{ width: '45%', marginRight: '0px', marginBottom: '0px' }}>
                                <div className='content'>
                                    <div className='sub-content'>
                                        <span className='key'>Average Fare</span>
                                        <span className='value'><i className='fa fa-arrow-up'></i>0.2M (1200)</span>
                                    </div>
                                    <div className='sub-content'>
                                        <span className='key'>Traffic Mix</span>
                                        <span className='value'><i className='fa fa-arrow-up'></i>0.1M (800)</span>
                                    </div>
                                    <span>(POS,Compartment & OD)</span>
                                </div>
                            </div>
                            <div className='inner-rounded-box' style={{ marginBottom: '0px' }}>
                                <div className='top-rb'>
                                    <p>Average Fare Movement</p>
                                </div>
                                <div className='content'>
                                    <div className='sub-content'>
                                        <span className='key' style={{ fontSize: '14px' }}>Components</span>
                                        <span className='value' style={{ fontSize: '14px' }}><p>Var</p>Var%</span>
                                    </div>
                                    <div className='sub-content'>
                                        <span className='key remove-margin'>Average Fare Increase</span>
                                        <span className='value'><p className='remove-margin'>120</p>30</span>
                                    </div>
                                    <div className='sub-content'>
                                        <span className='key remove-margin'>Exchange Rate</span>
                                        <span className='value'><p className='remove-margin'>120</p>20</span>
                                    </div>
                                    <div className='sub-content'>
                                        <span className='key remove-margin'>Traffic Mbc</span>
                                        <span className='value'><p className='remove-margin'>120</p>20</span>
                                    </div>
                                    <div className='sub-content'>
                                        <span className='key remove-margin'>Average Fare</span>
                                        <span className='value'><p className='remove-margin'>120</p>20</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div >
        )
    }

    render() {
        const { revenue, yaxisname, categories, dataset, loading } = this.state;
        return (
            <div>
                <Loader />
                <TopMenuBar dashboard={false} getPOSSelectedGeographicalDetails={this.getPOSSelectedGeographicalDetails} {...this.props} />
                <div className='targetRational'>
                    <div className='add'>
                        <h2>Target Rational</h2>
                    </div>
                    <div className='targetRational-form'>
                        {this.renderSection1()}
                        {this.renderSection2()}
                        {this.renderSection3()}
                    </div>
                </div>
            </div>
        );
    }
}

export default TargetRational;
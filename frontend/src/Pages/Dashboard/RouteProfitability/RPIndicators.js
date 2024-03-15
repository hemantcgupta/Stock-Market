import React, { Component } from "react";
import APIServices from '../../../API/apiservices';
import Spinners from "../../../spinneranimation";
import constant from '../../../Constants/validator'

const apiServices = new APIServices();

class Indicators extends Component {

    constructor(props) {
        super(props);
        this.state = {
            indicatorsDatas: [],
            indicatorsDivWidth: "",
            loading: false
        };
    }

    componentWillReceiveProps = (props) => {
        this.getIndicatorsData(props)
    }

    getIndicatorsData = (props) => {
        var self = this;
        const { startDate, endDate, routeGroup, regionId, countryId, cityId, routeId, dashboard } = props;
        self.setState({ loading: true, indicatorsDatas: [] })
        apiServices.getIndicatorsData(startDate, endDate, routeGroup, regionId, countryId, cityId, routeId, dashboard).then((data) => {
            self.setState({ loading: false })
            if (data) {
                self.setState({
                    indicatorsDatas: data
                })
            }
        });
    }

    redirectToDDSChart() {
        this.props.history.push(`/DDSChart`)
    }


    render() {
        const { dashboard } = this.props;
        const { indicatorsDatas, loading } = this.state;
        return (
            loading ?
                <div className='no-indicators'><Spinners /></div> :
                indicatorsDatas.length === 0 ?
                    <h4 className='no-indicators'>No Indicators to show</h4> :
                    <div>
                        {indicatorsDatas.map((data, i) => {
                            return (
                                <div id="indicator_data" className="col-sm-4 col-xs-6 tile_stats_count" >
                                    <div className='indicator-top'>
                                        <div className="count_top"> {data.Name} </div>
                                        <div className="count" id={data.Name}> {window.numberFormat(data.CY)} </div>
                                    </div>
                                    <div className='indicator-bottom' >
                                        <span style={{ flex: '1' }}>{window.numberFormat(data.LY)} LY</span>
                                        <div className='indicator-2'>
                                            <i className={`${(data.Name).includes('C') ? constant.costCardsArrowIndicator(window.numberFormat(data.VLY)) : constant.cardsArrowIndicator(window.numberFormat(data.VLY))}`}></i>
                                            <span>{window.numberFormat(data.VLY)}% VLY</span>
                                        </div>
                                    </div>
                                    <div className='indicator-bottom' >
                                        <span style={{ flex: '1' }}>{window.numberFormat(data.Budget)} BGT</span>
                                        <div className='indicator-2'>
                                            <i className={`${(data.Name).includes('C') ? constant.costCardsArrowIndicator(window.numberFormat(data.VTG)) : constant.cardsArrowIndicator(window.numberFormat(data.VTG))}`}></i>
                                            <span >{window.numberFormat(data.VTG)}% {`${dashboard === 'Route Profitability' ? 'VBGT' : 'VTG'}`}</span>
                                        </div>
                                    </div>
                                    {/* <div className="progress active" id="indicator_progress" style={{ "width": "99%" }}>
                                    <div className="progress-bar bg-success" role="progressbar" aria-valuenow="70" aria-valuemin="0" aria-valuemax="100" style={{ "width": data.Percentage + "%" }} >
                                    </div>
                                </div> */}
                                </div>
                            )
                        })}
                        {this.props.dashboard === 'Pos' ? <button className='btn dds-chart-btn search' onClick={() => this.redirectToDDSChart()}>DDS CHART</button> : ''}
                    </div>

        )

    }
}

export default Indicators;

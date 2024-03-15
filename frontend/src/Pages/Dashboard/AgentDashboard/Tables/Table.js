import React, { Component } from "react";
import TopAgentOD from './TopAgent&OD';
import Channel from './Channel';
import '../AgentDashboard.scss';

class Tables extends Component {
  constructor(props) {
    super(props);
    this.state = {
      currency: 'BC',
      country: 'Null'
    }
  }

  componentWillReceiveProps = (newProps) => {
    if (newProps.countryId !== this.props.countryId) {
      this.setState({ country: newProps.countryId },
        () => {
          if (this.state.country === 'Null') {
            this.setState({ currency: 'BC' })
          }
        })
    }
  }

  toggle = (e) => {
    let currency = e.target.value;
    this.setState({ currency: currency })
  }

  render() {
    return (
      <div className="x_panel">
        <div className="col-md-4 col-sm-4 col-xs-12 channel">
          <div className='currency-selection'>
            <h4>Select Currency :</h4>
            <select className="form-control cabinselect currency-dropdown" onChange={(e) => this.toggle(e)} disabled={this.state.country === 'Null'}>
              <option value='BC' selected={this.state.country === 'Null' || this.state.currency === 'BC' ? true : false}>BC</option>
              <option value='LC'>LC</option>
            </select>
          </div>
          <Channel
            gettingYear={this.props.gettingYear}
            gettingMonth={this.props.gettingMonth}
            regionId={this.props.regionId}
            countryId={this.props.countryId}
            cityId={this.props.cityId}
            currency={this.state.currency}
          />
        </div>
        <div className="col-md-4 col-sm-4 col-xs-12 anciallary">
          <TopAgentOD
            gettingYear={this.props.gettingYear}
            gettingMonth={this.props.gettingMonth}
            regionId={this.props.regionId}
            countryId={this.props.countryId}
            cityId={this.props.cityId}
            currency={this.state.currency}
          />
        </div>
      </div>
    )
  }
}
export default Tables;